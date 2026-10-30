# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841
# [MODULE DOC] Tests for TorchScript graph-mode (JIT) quantization: the passes and APIs
# prepare_jit, convert_jit, quantize_jit, their dynamic counterparts and fuse_conv_bn_jit. Five
# test classes: TestQuantizeJitPasses (individual passes checked with FileCheck and observer
# counts), TestQuantizeJitOps (per-operator static quantization end to end),
# TestQuantizeDynamicJitPasses and TestQuantizeDynamicJitOps (dynamic quantization), and
# TestQuantizeJit (numeric comparison against eager-mode quantization). Importing has no side
# effects beyond imports. Executing the file directly calls raise_on_run_directly, which points the
# user to test/test_quantization.py instead of running tests.

# torch
import io
import itertools
import unittest

import torch
import torch.jit
import torch.jit.quantized
import torch.nn as nn
import torch.nn.functional as F

# torch.ao.quantization
from torch.ao.quantization import (
    default_dynamic_qconfig,
    default_histogram_observer,
    default_observer,
    default_per_channel_weight_observer,
    default_qconfig,
    default_weight_observer,
    float16_dynamic_qconfig,
    fuse_modules,
    get_default_qconfig,
    per_channel_dynamic_qconfig,
    PlaceholderObserver,
    QConfig,
    quantize,
    quantize_dynamic,
    quantize_dynamic_jit,
    quantize_jit,
)

# torch.ao.quantization.quantize_jit
from torch.ao.quantization.quantize_jit import (
    convert_dynamic_jit,
    convert_jit,
    fuse_conv_bn_jit,
    prepare_dynamic_jit,
    prepare_jit,
    script_qconfig,
)
from torch.jit._recursive import wrap_cpp_module
from torch.testing import FileCheck

# Annotated models
from torch.testing._internal.common_quantization import (
    AnnotatedConvBnModel,
    AnnotatedConvModel,
    AnnotatedConvTransposeModel,
    AnnotatedNestedModel,
    AnnotatedSingleLayerLinearModel,
    AnnotatedSkipQuantModel,
    ConvBnModel,
    ConvModel,
    ConvTransposeModel,
    default_per_channel_qconfig,
    get_script_module,
    NestedModel,
    QuantizationTestCase,
    SingleLayerLinearModel,
    skipIfNoFBGEMM,
    SkipQuantModel,
    test_only_eval_fn,
)

# Testing utils
from torch.testing._internal.common_quantized import (
    override_qengines,
    qengine_is_fbgemm,
    qengine_is_qnnpack,
)
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    set_default_dtype,
)
from torch.testing._internal.jit_utils import (
    attrs_with_prefix,
    get_forward,
    get_forward_graph,
)


# Standard library


class TestQuantizeJitPasses(QuantizationTestCase):
    # [CLASS DOC] Test case for the individual TorchScript graph-mode quantization passes behind
    # quantize_jit: conv-batchnorm folding, linear fusion, observer insertion, quant/dequant
    # insertion, dequantize replication, module-use dedup and finalization. Each test builds a
    # small module, scripts or traces it, runs a pass through prepare_jit, convert_jit or a raw
    # torch._C pass, and asserts on observer attributes, FileCheck graph patterns or numerics.
    # Inherits the shared quantization fixtures (such as img_data_2d) from QuantizationTestCase.
    """Test graph mode quantization passes used by quantize_jit"""

    def test_skip_dequant_constant_prop(self):
        # [FUNCTION DOC] Checks that constant propagation after freezing skips over dequantize
        # nodes. Scripts a single-conv module, prepares it with a per-channel weight observer and a
        # per-tensor activation observer, calibrates on one input, converts with debug on, then
        # freezes and runs it. Asserts on the frozen graph: exactly 2 per-tensor quantize nodes, no
        # per-channel quantize node, 3 dequantize nodes, and the expected node order, where the
        # weight remains a constant feeding dequantize directly.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Float module holding one Conv2d (3 in, 5 out, kernel 3) and
            # applying it to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the float Conv2d submodule named conv.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the conv applied to x.
                return self.conv(x)

        m = torch.jit.script(M())
        observer = default_per_channel_weight_observer.with_args(ch_axis=1)
        qconfig_dict = {"": QConfig(activation=default_observer, weight=observer)}
        m = prepare_jit(m, qconfig_dict)
        data = torch.randn(1, 3, 10, 10, dtype=torch.float)

        m(data)
        m = convert_jit(m, debug=True)

        freezed = torch.jit.freeze(m)
        freezed(data)

        # After freezing, weight becomes Constant.
        # We have this pattern in the original graph: Constant f32_weight -> quant -> dequant
        # After skipping dequant during Constant Propagation, the resulting graph will be:
        # Constant int8_weight -> dequant
        FileCheck().check_count("aten::quantize_per_tensor", 2, exactly=True).run(
            freezed.graph
        )
        FileCheck().check_count("aten::quantize_per_channel", 0, exactly=True).run(
            freezed.graph
        )
        FileCheck().check_count("aten::dequantize", 3, exactly=True).run(freezed.graph)
        FileCheck().check("aten::quantize_per_tensor").check_next(
            "aten::dequantize"
        ).check_not("aten::quantize_per_channel").check("aten::dequantize").check_next(
            "aten::conv2d"
        ).check_next("aten::quantize_per_tensor").check_next("aten::dequantize").run(
            freezed.graph
        )

    def test_foldbn_trivial(self):
        # [FUNCTION DOC] Checks that fuse_conv_bn_jit folds a BatchNorm directly following a Conv
        # without changing numerics. For both scripted and traced modules and for 2d and 3d, builds
        # a conv+bn eval module (bn eps changed from default), asserts that two forward call-method
        # nodes exist before the pass and one after, and compares the fused module output with the
        # eager output using assertEqual.
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # Test trivial case
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conv and a batchnorm of the requested
            # dimensionality (2 or 3, chosen via the enclosing dictionaries), applied in sequence;
            # the batchnorm eps is set to a non-default value.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Builds conv (1 to 20 channels, kernel 5, stride 1) and bn
                # (20 features) for dimension dim, and overrides the bn eps.
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1)
                self.bn = bn_module[dim](num_features=20)
                self.bn.eps = 0.0023

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then bn and returns the result.
                x = self.conv(x)
                x = self.bn(x)
                return x

        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 6, 6), 3: torch.rand(1, 1, 6, 6, 6)}
        # Check that the transformation doesn't change numerics
        for tracing, dim in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            # Check that in the original script module's forward we have two
            # CallMethod nodes. One of them should be for conv.forward and the other
            # for bn.forward.
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward(scripted_or_traced._c).graph))

            # Run FoldConvBatchnorm pass.
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            # Check that after the pass one of the CallMethods is gone (supposedly,
            # the bn.forward).
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # Check that the transformation doesn't change numerics
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_trivial_nobias(self):
        # [FUNCTION DOC] Same as the trivial fold test but the conv has no bias. The bn has a
        # non-default eps and a random bias so the folded bias is nonzero. For scripted and traced
        # 2d/3d modules, asserts two forward call-method nodes before fuse_conv_bn_jit and one
        # after, and that outputs still match eager.
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # Test trivial case
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv (bias disabled) followed by batchnorm; bn gets a non-default
            # eps and a random trainable bias so the fused bias is nonzero.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Builds a bias-free conv and a bn of dimension dim, sets bn
                # eps and replaces bn bias with a random parameter.
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1, bias=False)
                self.bn = bn_module[dim](num_features=20)
                # to make sure new bias is not zero
                self.bn.eps = 0.0027
                self.bn.bias = torch.nn.Parameter(torch.rand([20]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then bn and returns the result.
                x = self.conv(x)
                x = self.bn(x)
                return x

        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 6, 6), 3: torch.rand(1, 1, 6, 6, 6)}
        for tracing, dim in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            # Check that in the original script module's forward we have two
            # CallMethod nodes. One of them should be for conv.forward and the other
            # for bn.forward.
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # Run FoldConvBatchnorm pass.
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            # Check that after the pass one of the CallMethods is gone (supposedly,
            # the bn.forward).
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # Check that the transformation doesn't change numerics
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_in_submodule(self):
        # [FUNCTION DOC] Checks that conv-bn patterns inside a submodule are found and folded.
        # Wraps a conv+bn submodule in a parent, scripts or traces it for 2d and 3d, asserts the
        # submodule graph goes from two forward call-method nodes to one after fuse_conv_bn_jit,
        # and that outputs match eager.
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # Test that we find Conv-BN patterns in submodules
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule holding a conv and a batchnorm of dimension dim,
            # applied in order.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Builds the conv (1 to 20 channels, kernel 5) and bn (20
                # features) for dimension dim.
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1)
                self.bn = bn_module[dim](num_features=20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then bn and returns the result.
                x = self.conv(x)
                x = self.bn(x)
                return x

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module that owns one SubModule as attribute sub and
            # delegates to it.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates the SubModule attribute sub for dimension dim.
                super().__init__()
                self.sub = SubModule(dim)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the output of the sub module on x.
                x = self.sub(x)
                return x

        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 10, 10), 3: torch.rand(1, 1, 10, 10, 10)}
        for tracing, dim in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub._c)))

            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub._c)))

            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_shared_classtype(self):
        # [FUNCTION DOC] Checks that folding works when two conv/bn pairs share a class type in a
        # scripted or traced module. Runs 2d only (the dimension list repeats 2) with and without
        # conv bias and tracing, calls fuse_conv_bn_jit, and compares eager output with the output
        # of the module passed to the pass.
        # KNOWN ISSUE: the compared output is from the original scripted_or_traced module, not the
        # returned folded module, so it only checks the pass did not break the input in place.
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Two conv+bn+relu blocks (conv1/bn1, conv2/bn2, one shared relu).
            # bn1 has a modified running mean, random bias and eps; bn2 has a different eps, so
            # blocks differ in statistics but share class types.
            def __init__(self, dim, bias=False):
                # [FUNCTION DOC] HELPER: Builds both conv layers (5 to 5 channels, kernel 3, bias
                # optional), both bn layers of dimension dim, and sets the bn statistics, bias and
                # eps values.
                super().__init__()
                self.conv1 = conv_module[dim](5, 5, 3, bias=bias)
                self.bn1 = bn_module[dim](num_features=5)
                self.bn1.running_mean.fill_(-0.2)
                self.bn1.bias = torch.nn.Parameter(torch.rand([5]))
                # to make sure new bias is not zero
                self.bn1.eps = 0.0023
                self.conv2 = conv_module[dim](5, 5, 3, bias=bias)
                self.bn2 = bn_module[dim](num_features=5)
                self.bn2.eps = 0.0029
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs conv1, bn1, relu, conv2, bn2, relu in order and
                # returns the result.
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        options = itertools.product([True, False], [2, 2], [True, False])
        data = {2: torch.rand(1, 5, 6, 6), 3: torch.rand(1, 5, 6, 6, 6)}
        for tracing, dim, bias in options:
            eager = TestModule(dim, bias).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x)
            folded = fuse_conv_bn_jit(scripted_or_traced)
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_no_fusion(self):
        # [FUNCTION DOC] Checks that fuse_conv_bn_jit does not fuse when the submodules are custom
        # module types rather than real conv and batchnorm. Scripts a module of two identity-like
        # custom modules, applies the pass, and asserts the graph still has exactly two
        # call-method nodes.
        """Test that we don't fuse the cases when module type does not match"""

        class CustomConv(torch.nn.Module):
            # [CLASS DOC] HELPER: Custom module whose forward returns the input unchanged; stands
            # in for a non-conv module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x unchanged.
                return x

        class CustomBn(torch.nn.Module):
            # [CLASS DOC] HELPER: Custom module whose forward returns the input unchanged; stands
            # in for a non-batchnorm module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x unchanged.
                return x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module composed of CustomConv as conv and CustomBn as bn, applied
            # in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two custom submodules conv and bn.
                super().__init__()
                self.conv = CustomConv()
                self.bn = CustomBn()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Feeds x through conv then bn and returns the result.
                return self.bn(self.conv(x))

        m = torch.jit.script(M())
        m = fuse_conv_bn_jit(m)
        FileCheck().check_count("prim::CallMethod", 2, exactly=True).run(m.graph)

    @set_default_dtype(torch.double)
    def test_foldbn_complex_cases(self):
        # [FUNCTION DOC] Exercises conv-bn folding over many combinations (2d/3d, conv bias or not,
        # bn affine or not, 1 or 2 blocks, scripted or traced) with the default dtype set to double
        # by the decorator. For each, asserts the number of forward call-method nodes in the layer
        # sequence is twice the block count before and equal to the block count after
        # fuse_conv_bn_jit, and that fused and eager outputs match.
        # This test case attempt to try combinations of conv2d/conv3d with bias/nobias
        # as well as BatchNorm with affine/no-affine along with varying the
        # number of layers.
        # this only works when default dtype is double
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a Sequential of num_blocks conv+bn pairs; bn layers get
            # random running statistics and, if affine, random weight and bias.
            def __init__(self, dim, num_blocks, enable_bias, enable_affine):
                # [FUNCTION DOC] HELPER: Builds conv layers (20 to 20 channels, kernel 5, optional
                # bias) alternating with bn layers whose running mean, running variance and (if
                # affine enabled) weight and bias are randomized, and stores them as the
                # Sequential layers.
                super().__init__()
                layers = []
                for _ in range(num_blocks):
                    layers.append(conv_module[dim](20, 20, 5, 1, bias=enable_bias))
                    bn_obj = bn_module[dim](num_features=20, affine=enable_affine)
                    if enable_affine:
                        bn_obj.weight = torch.nn.Parameter(
                            torch.rand_like(bn_obj.weight)
                        )
                        bn_obj.bias = torch.nn.Parameter(torch.rand_like(bn_obj.bias))
                    bn_obj.running_mean = torch.rand_like(bn_obj.running_mean)
                    bn_obj.running_var = torch.rand_like(bn_obj.running_var)
                    layers.append(bn_obj)
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the layers sequence applied to x.
                return self.layers(x)

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper module that owns one SubModule as sub.
            def __init__(self, dim, num_blocks, enable_bias, enable_affine):
                # [FUNCTION DOC] HELPER: Creates the SubModule attribute sub with the given
                # dimension, block count, bias and affine flags.
                super().__init__()
                self.sub = SubModule(dim, num_blocks, enable_bias, enable_affine)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies sub to x and returns the result.
                x = self.sub(x)
                return x

        options = itertools.product(
            [True, False], [2, 3], [True, False], [True, False], [1, 2]
        )
        data = {2: torch.rand(1, 20, 10, 10), 3: torch.rand(1, 20, 10, 10, 10)}
        for tracing, dim, enable_bias, enable_bn_affine, num_layers in options:
            eager = TestModule(dim, num_layers, enable_bias, enable_bn_affine).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()

            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', num_layers * 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))

            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', num_layers, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))

            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_fuse_linear(self):
        # [FUNCTION DOC] Checks the linear fusion pass torch._C._jit_pass_fuse_linear on traced
        # models. For three input shapes (1d, 2d, 3d input) with and without bias, traces a
        # matmul-with-transposed-weight plus optional in-place bias add, runs the pass, asserts the
        # graph has a linear op and no matmul, addmm, add_ or transpose, that the source range of
        # the fused node equals that of the original matmul, and that the model runs. Then checks a
        # 3d-by-3d matmul without transpose is not fused into linear.
        class FunctionalLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Functional linear built from matmul with the transposed weight
            # and an in-place add of bias when bias is not None. Weight and bias are stored as
            # plain attributes.
            def __init__(self, weight, bias):
                # [FUNCTION DOC] HELPER: Stores the given weight and bias tensors (bias may be
                # None) as attributes.
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes matmul of x with the transposed weight; adds bias
                # in place if present; returns the result.
                res = torch.matmul(x, self.weight.t())
                if self.bias is not None:
                    res.add_(self.bias)
                return res

        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)

        x2 = torch.rand(5, 5)
        w2 = torch.rand(5, 5)
        b2 = torch.rand(5)

        x3 = torch.rand(5, 5, 5)
        w3 = torch.rand(5, 5)
        b3 = torch.rand(5)
        for has_bias, (x, weight, b) in itertools.product(
            [True, False], [(x1, w1, b1), (x2, w2, b2), (x3, w3, b3)]
        ):
            bias = b if has_bias else None
            model = torch.jit.trace(FunctionalLinear(weight, bias), [x])
            for node in model.graph.nodes():
                if node.kind() == "aten::matmul":
                    source_range_1 = node.sourceRange()
            torch._C._jit_pass_fuse_linear(model.graph)
            for node in model.graph.nodes():
                if node.kind() == "aten::linear":
                    source_range_2 = node.sourceRange()
            FileCheck().check("aten::linear").run(model.graph)
            check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            # make sure it runs
            self.assertTrue(source_range_1 == source_range_2)
            model(x)

        # check matmuls are not fused
        class Matmul(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that returns the plain matmul of the input and a stored
            # weight (no transpose), used to check matmul is not fused.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Stores the weight tensor as an attribute.
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matmul of x and the stored weight.
                return torch.matmul(x, self.weight)

        x = torch.rand(5, 6, 5)
        w = torch.rand(5, 5, 100)
        model = torch.jit.trace(Matmul(w), [x])
        torch._C._jit_pass_fuse_linear(model.graph)
        # check 3d matmul is not fused
        FileCheck().check("aten::matmul").run(model.graph)
        FileCheck().check_not("aten::linear").run(model.graph)
        # make sure it runs
        model(x)

    def test_insert_observers(self):
        # [FUNCTION DOC] Checks basic observer insertion. Scripts a single-conv module, runs
        # prepare_jit with the default qconfig for the root, and raises an error unless the root
        # module has exactly 2 observer attributes (conv input and output) and the conv has exactly
        # 1 (its weight).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single Conv2d (3 to 5, kernel 3) applied to
            # the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Conv2d attribute conv.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv applied to x.
                return self.conv(x)

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # for input and output of conv
        if len(attrs_with_prefix(m, "_observer_")) != 2:
            raise AssertionError(
                f"Expected 2 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
            )
        # for weight
        if len(attrs_with_prefix(m.conv, "_observer_")) != 1:
            raise AssertionError(
                f"Expected 1 observer, got {len(attrs_with_prefix(m.conv, '_observer_'))}"
            )

    def test_insert_observers_interface(self):
        # [FUNCTION DOC] Smoke test that prepare_jit handles a module that has a TorchScript
        # interface-typed class declared. Scripts a conv plus Sub module and calls prepare_jit with
        # a qconfig keyed on a submodule name; nothing is asserted beyond not raising.
        @torch.jit.interface
        class SubInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: TorchScript interface declaring the addOne method that returns a
            # tensor; the declared body only passes. It is not referenced by M or Sub in this test.
            def addOne(self, inp) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no implementation.
                pass

        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear layer; forward delegates to addOne, which
            # adds one to the linear output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def addOne(self, inp):
                # [FUNCTION DOC] HELPER: Returns the fc output plus one.
                return self.fc(inp) + 1

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the result of addOne on x.
                return self.addOne(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module composing a conv and a Sub submodule, applied conv first
            # then sub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the conv (3 to 5, kernel 3) and the Sub submodule.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then sub and returns the result.
                return self.sub(self.conv(x))

        m = torch.jit.script(M())
        qconfig_dict = {"sub.conv": default_qconfig}
        m = prepare_jit(m, qconfig_dict)

    def test_insert_observers_interface_unshare_type(self):
        # [FUNCTION DOC] Checks that prepare_jit works when two submodules of the same
        # interface-typed attribute hold different concrete types (Operator with int vs float
        # constant). For both traced and scripted versions of an outer module, runs prepare_jit
        # with qconfig entries for both inner modules; nothing is asserted beyond not raising.
        @torch.jit.interface
        class OperatorIf(nn.Module):
            # [CLASS DOC] HELPER: TorchScript interface for a module whose forward maps a tensor to
            # a tensor; body only passes.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface forward declaration with no implementation.
                pass

        class Operator(nn.Module):
            # [CLASS DOC] HELPER: Module that stores a constant a and computes a times the sum of
            # input and a.
            def __init__(self, a):
                # [FUNCTION DOC] HELPER: Stores the number a (int or float) as an attribute.
                super().__init__()
                self.a = a

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns a times (inp plus a).
                return self.a * (inp + self.a)

        class Inner(nn.Module):
            # [CLASS DOC] HELPER: Wrapper declaring an attribute op annotated with the OperatorIf
            # interface type and forwarding input to it.
            op: OperatorIf

            def __init__(self, op):
                # [FUNCTION DOC] HELPER: Stores the given op module as attribute op.
                super().__init__()
                self.op = op

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns op applied to inp.
                return self.op(inp)

        class Outer(nn.Module):
            # [CLASS DOC] HELPER: Module with two Inner wrappers whose operators have different
            # constant types (integer 1 and float 3.0); forward sums their outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates inner_a with Operator(1) and inner_b
                # with Operator(3.0).
                super().__init__()
                self.inner_a = Inner(Operator(1))
                self.inner_b = Inner(Operator(3.0))

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the sum of inner_a and inner_b applied to inp.
                return self.inner_a(inp) + self.inner_b(inp)

        qconfig_dict = {"inner_a": default_qconfig, "inner_b": default_qconfig}

        eager_model = Outer()
        for tracing in [True, False]:
            x = torch.rand(3)
            script_model = get_script_module(eager_model, tracing, x)
            # make sure it runs
            prepare_jit(script_model, qconfig_dict)

    def test_insert_observers_child_qconfig(self):
        # [FUNCTION DOC] Checks which modules receive observers when only a nested submodule has a
        # qconfig. Uses a qconfig only for sub.fc; raises an error unless the root has 2 observers
        # (input and output of sub), conv has 0, sub has 0 (observation happens at the outermost
        # call site) and the Linear fc has 1 (weight).
        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a single Linear layer named fc.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc applied to x.
                return self.fc(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conv followed by a Sub submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv (3 to 5, kernel 3) and the Sub submodule.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then sub and returns the result.
                return self.sub(self.conv(x))

        m = torch.jit.script(M())
        qconfig_dict = {"sub.fc": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # input and output of sub
        if len(attrs_with_prefix(m, "_observer_")) != 2:
            raise AssertionError(
                f"Expected 2 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
            )
        # not quantized
        if len(attrs_with_prefix(m.conv, "_observer_")) != 0:
            raise AssertionError(
                f"Expected 0 observers, got {len(attrs_with_prefix(m.conv, '_observer_'))}"
            )
        # no observers since we observe in the outer most call site
        if len(attrs_with_prefix(m.sub, "_observer_")) != 0:
            raise AssertionError(
                f"Expected 0 observers, got {len(attrs_with_prefix(m.sub, '_observer_'))}"
            )
        # weight of linear
        if len(attrs_with_prefix(m.sub.fc, "_observer_")) != 1:
            raise AssertionError(
                f"Expected 1 observer, got {len(attrs_with_prefix(m.sub.fc, '_observer_'))}"
            )

    @unittest.skipUnless(
        "fbgemm" in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_insert_observers_skip_values(self):
        # [FUNCTION DOC] Checks that observers are skipped for values that will be fused with an op
        # (conv followed by relu, add followed by relu). Counts observer attributes after
        # prepare_jit for four modules, and for the add modules uses FileCheck to confirm no
        # observer is inserted between the in-place add and the relu. Only runs when the fbgemm
        # engine is supported (skip decorator).
        class ConvFunctionalReLU(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by the functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Conv2d (3 to 5, kernel 3) as conv.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional relu of the conv output.
                return F.relu(self.conv(x))

        class ConvReLUModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by a ReLU module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv (3 to 5, kernel 3) and a ReLU module.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu module applied to the conv output.
                return self.relu(self.conv(x))

        class AddReLUModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv plus an in-place residual add of the input, then a
            # ReLU module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a ReLU module and a float conv (3 to 3, kernel 3).
                super().__init__()
                self.relu = torch.nn.ReLU()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes conv of x, adds x in place, and returns relu
                # module applied to the sum.
                out = self.conv(x)
                out += x
                return self.relu(out)

        class AddFunctionalReLU(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv plus an in-place residual add of the input, then the
            # functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a float conv (3 to 3, kernel 3).
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes conv of x, adds x in place, and returns
                # functional relu of the sum.
                out = self.conv(x)
                out += x
                return F.relu(out)

        def attrs_with_prefix(module, prefix):
            # [FUNCTION DOC] HELPER: Local version of the attribute lister that shadows the
            # imported one: returns the names of submodules of module whose names start with
            # prefix, by reading the underlying C++ module children.
            return [x for x, _ in module._modules._c.items() if x.startswith(prefix)]

        qconfig_dict = {"": default_qconfig}
        m = torch.jit.script(ConvFunctionalReLU())
        m = prepare_jit(m, qconfig_dict)
        # observer for weight of conv
        if len(attrs_with_prefix(m.conv, "_observer_")) != 1:
            raise AssertionError(
                f"Expected 1 observer, got {len(attrs_with_prefix(m.conv, '_observer_'))}"
            )
        # observer for input of conv and output of relu
        if len(attrs_with_prefix(m, "_observer_")) != 2:
            raise AssertionError(
                f"Expected 2 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
            )

        m = torch.jit.script(ConvReLUModule())
        m = prepare_jit(m, qconfig_dict)
        # observer for input of conv and output of relu
        if len(attrs_with_prefix(m, "_observer_")) != 2:
            raise AssertionError(
                f"Expected 2 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
            )
        # observer for weight of conv
        if len(attrs_with_prefix(m.conv, "_observer_")) != 1:
            raise AssertionError(
                f"Expected 1 observer, got {len(attrs_with_prefix(m.conv, '_observer_'))}"
            )
        # observer for output of relu
        if len(attrs_with_prefix(m.relu, "_observer_")) != 0:
            raise AssertionError(
                f"Expected 0 observers, got {len(attrs_with_prefix(m.relu, '_observer_'))}"
            )

        m = torch.jit.script(AddReLUModule())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        if len(attrs_with_prefix(m, "_observer")) != 3:
            raise AssertionError(
                f"Expected 3 observers, got {len(attrs_with_prefix(m, '_observer'))}"
            )
        if len(attrs_with_prefix(m.relu, "_observer")) != 0:
            raise AssertionError(
                f"Expected 0 observers, got {len(attrs_with_prefix(m.relu, '_observer'))}"
            )
        FileCheck().check("aten::add_").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("ReLU = prim::GetAttr").run(str(get_forward_graph(m._c)))

        m = torch.jit.script(AddFunctionalReLU())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        if len(attrs_with_prefix(m, "_observer")) != 3:
            raise AssertionError(
                f"Expected 3 observers, got {len(attrs_with_prefix(m, '_observer'))}"
            )
        FileCheck().check("aten::add_").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("CallFunction").check('Observer = prim::GetAttr[name="_observer_').run(
            str(get_forward_graph(m._c))
        )

    def test_insert_observers_weight_dtype(self):
        # [FUNCTION DOC] Checks that observer attributes of activations and of the conv weight use
        # different dtypes. Collects the dtype attribute of observers on the root and on conv,
        # requires exactly one distinct dtype for each, and requires the two dtypes to differ.
        # KNOWN ISSUE: the message of the last raise is truncated (it ends at the word to) and the
        # rest of the intended text is a separate no-effect string statement afterwards.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conv followed by functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Conv2d (3 to 5, kernel 3) as conv.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional relu of the conv output.
                return F.relu(self.conv(x))

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        activation_dtypes = {
            obs.getattr("dtype")
            for x, obs in m._modules._c.items()
            if x.startswith("_observer_")
        }
        weight_dtypes = {
            obs.getattr("dtype")
            for x, obs in m.conv._modules._c.items()
            if x.startswith("_observer_")
        }
        if len(activation_dtypes) != 1:
            raise AssertionError("Expected to have 1 activation dtype")
        if len(weight_dtypes) != 1:
            raise AssertionError("Expected to have 1 weight dtype")
        if next(iter(activation_dtypes)) == next(iter(weight_dtypes)):
            raise AssertionError("Expected activation dtype to ")
        " be different from wegiht dtype"

    def test_insert_observers_for_reused_weight(self):
        # [FUNCTION DOC] Checks observer count when one weight tensor is passed to two functional
        # convolutions. Scripts a module taking x, y and weight, prepares it, and requires 6
        # observer attributes: one each for x, y and weight, one per conv output and one for the
        # add output.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no parameters that applies functional conv2d to two
            # inputs with the same externally supplied weight and adds the results.
            def forward(self, x, y, weight):
                # [FUNCTION DOC] HELPER: Returns conv2d of x with weight plus conv2d of y with the
                # same weight.
                x = F.conv2d(x, weight)
                y = F.conv2d(y, weight)
                return x + y

        m = torch.jit.script(M()).eval()
        m = prepare_jit(m, {"": default_qconfig})
        # 3 for x, y, weight, one for output of each F.conv2d and one for output of add
        if len(attrs_with_prefix(m, "_observer")) != 6:
            raise AssertionError(
                f"Expected 6 observers, got {len(attrs_with_prefix(m, '_observer'))}"
            )

    def test_insert_observers_shared_class_type(self):
        # [FUNCTION DOC] Checks that two convs sharing a class type are observed independently of
        # the type but get identical observer attribute names. After prepare_jit, requires each
        # conv to have exactly one observer and the two name lists to be equal, showing the shared
        # type was only quantized once.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two float convs (3 to 5, kernel 3), conv1
            # feeding conv2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two convs conv1 and conv2.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 5, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv2 applied to the output of conv1.
                return self.conv2(self.conv1(x))

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # conv1 and conv2 shares the same type, we need to
        # make sure we didn't quantize the type twice
        conv1_observers = attrs_with_prefix(m.conv1, "_observer_")
        conv2_observers = attrs_with_prefix(m.conv2, "_observer_")
        if len(conv1_observers) != 1:
            raise AssertionError("Expected to have 1 observer submodules")
        if len(conv2_observers) != 1:
            raise AssertionError("Expected to have 1 observer submodules")
        if conv1_observers != conv2_observers:
            raise AssertionError(
                "Expect conv1 and conv2 to have same observers since the class type is shared"
            )

    def test_insert_observers_for_general_ops(self):
        # [FUNCTION DOC] Checks observers are not inserted after ops that do not need observation
        # (flatten). Prepares conv plus flatten, requires 2 observers on the root, and uses
        # FileCheck to confirm there is no observer after the flatten op.
        """Make sure we skip observers for ops that doesn't require
        observation, e.g. flatten
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by flatten.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 3).
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the flattened conv output.
                x = self.conv(x)
                x = torch.flatten(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # input and output of conv
        if len(attrs_with_prefix(m, "_observer_")) != 2:
            raise AssertionError(
                f"Expected 2 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
            )
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("aten::flatten").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(m.graph)

    # TODO: this is too long, split this to test_insert_observers.py and remove
    # insrt_observers prefix
    def test_insert_observers_propagate_observed(self):
        # [FUNCTION DOC] Checks that the observed property propagates through a general op
        # (flatten) so the next conv input needs no extra observer. Requires 3 observers on the
        # root and uses FileCheck to confirm an observer after conv1, none after the flatten, and
        # an observer after conv2.
        """Make sure we propagate observed property through general ops"""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: conv1, then flatten, then conv2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (3 to 3, kernel 3).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv1, flatten, then conv2 and returns the result.
                x = self.conv1(x)
                x = torch.flatten(x)
                # we don't want to insert observer for input of self.conv2
                # because output of self.conv1 is already observed
                x = self.conv2(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # input and output of conv
        if len(attrs_with_prefix(m, "_observer_")) != 3:
            raise AssertionError(
                f"Expected 3 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
            )
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv1"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("aten::flatten").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check('prim::GetAttr[name="conv2"]').check(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(m.graph)

    def test_insert_observers_propagate_observed_in_submodule(self):
        # [FUNCTION DOC] Same as the propagation test but with an adaptive average pool submodule
        # between the convs. Requires 3 observers on the root and FileCheck verifies no observer is
        # inserted between pooling and conv2.
        """Make sure we propagate observed property through general ops"""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: conv1, then adaptive average pool to 1 by 1, then conv2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (3 to 3, kernel 3) and an
                # AdaptiveAvgPool2d with output size 1 by 1.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv1, avgpool, then conv2 and returns the result.
                x = self.conv1(x)
                x = self.avgpool(x)
                # we don't want to insert observer for input of self.conv2
                # because output of self.conv1 is already observed
                x = self.conv2(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # input and output of conv
        if len(attrs_with_prefix(m, "_observer_")) != 3:
            raise AssertionError(
                f"Expected 3 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
            )
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv1"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("prim::CallMethod").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check('prim::GetAttr[name="conv2"]').check(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(m.graph)

    def test_insert_observers_propagate_observed_for_function(self):
        # [FUNCTION DOC] Checks that observed property propagates through a plain Python function
        # (channel shuffle) called from forward. Prepares a scripted module with that function
        # between two convs and requires 3 observers on the root, so the input of conv2 is not
        # observed again. A calibration data list is built but not used.
        def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Channel shuffle: reads the batch, channel, height and width
            # sizes, splits the channels into groups, transposes group and channel-per-group
            # dimensions, and flattens back; returns the shuffled tensor.
            batchsize, num_channels, height, width = x.data.size()
            channels_per_group = num_channels // groups
            # reshape
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            # flatten
            x = x.view(batchsize, -1, height, width)
            return x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: conv1, then channel_shuffle with one group, then conv2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (3 to 3, kernel 1).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv1, channel_shuffle with one group, conv2 and
                # returns the result.
                x = self.conv1(x)
                x = channel_shuffle(x, 1)
                x = self.conv2(x)
                return x

        data = [
            (
                torch.rand((1, 3, 10, 10), dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            )
            for _ in range(2)
        ]
        m = torch.jit.script(M()).eval()
        m = prepare_jit(m, {"": default_qconfig})
        # we want to test that channel_shuffle is going to pass
        # the observed property from the output of conv1 to input of conv2
        # so that we don't insert observers for input of conv2
        if (
            len(
                attrs_with_prefix(
                    m,
                    "_observer_",
                )
            )
            != 3
        ):
            raise AssertionError(
                f"Expected 3 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
            )

    def test_insert_observers_for_if(self):
        # [FUNCTION DOC] Checks observer placement around conditionals. Uses a trace and a script
        # path for a model made of a quant-propagating module (branches ending in reshape) and a
        # residual module (branches ending in conv). Expected observer counts for root,
        # quant_prop and res are 2, 1, 0 when traced and 1, 2, 2 when scripted; raises an error
        # on any mismatch.
        class QuantProp(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conv and a boolean use_skip; both branches of the
            # conditional apply conv then reshape to the same shape.
            def __init__(self, use_skip):
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 1) and stores the
                # use_skip flag.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = use_skip

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then reshape in whichever branch use_skip
                # selects (both identical) and returns the result.
                if self.use_skip:
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)
                else:
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)

        class Res(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conv and boolean use_skip; both branches return the
            # conv output.
            def __init__(self, use_skip):
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 1) and stores the
                # use_skip flag.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = use_skip

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv applied to x in either branch.
                if self.use_skip:
                    return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Chain of QuantProp(True) as quant_prop followed by Res(False)
            # as res.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates quant_prop and res submodules.
                super().__init__()
                self.quant_prop = QuantProp(True)
                self.res = Res(False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies quant_prop then res and returns the result.
                x = self.quant_prop(x)
                x = self.res(x)
                return x

        data = [torch.rand(1, 3, 10, 10, dtype=torch.float)]
        result = {False: [1, 2, 2], True: [2, 1, 0]}
        for tracing in [True, False]:
            if tracing:
                m = torch.jit.trace(M(), data).eval()
            else:
                m = torch.jit.script(M()).eval()
            m = prepare_jit(m, {"": default_qconfig})
            if (
                len(
                    attrs_with_prefix(
                        m,
                        "_observer_",
                    )
                )
                != result[tracing][0]
            ):
                raise AssertionError(
                    f"Expected {result[tracing][0]} observers on m, "
                    f"got {len(attrs_with_prefix(m, '_observer_'))}"
                )
            if (
                len(
                    attrs_with_prefix(
                        m.quant_prop,
                        "_observer_",
                    )
                )
                != result[tracing][1]
            ):
                raise AssertionError(
                    f"Expected {result[tracing][1]} observers on m.quant_prop, "
                    f"got {len(attrs_with_prefix(m.quant_prop, '_observer_'))}"
                )
            if (
                len(
                    attrs_with_prefix(
                        m.res,
                        "_observer_",
                    )
                )
                != result[tracing][2]
            ):
                raise AssertionError(
                    f"Expected {result[tracing][2]} observers on m.res, "
                    f"got {len(attrs_with_prefix(m.res, '_observer_'))}"
                )

    def test_insert_observers_for_nested_if(self):
        # [FUNCTION DOC] Checks observer count for nested conditionals. Traced versions are
        # expected to have 3 root observers and scripted versions 1; raises an error otherwise.
        class Res(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a conv and two boolean attributes (cond and use_skip)
            # used for a nested conditional whose three leaves all return the conv output.
            def __init__(self, use_skip):
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 1) and stores use_skip
                # in both cond and use_skip.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.cond = use_skip
                self.use_skip = use_skip

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv of x from whichever nested branch is taken
                # (all identical).
                if self.use_skip:
                    if self.cond:
                        return self.conv(x)
                    else:
                        return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two Res modules in sequence: res1 built with True and res2
            # with False.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates res1 and res2.
                super().__init__()
                self.res1 = Res(True)
                self.res2 = Res(False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies res1 then res2 and returns the result.
                x = self.res1(x)
                x = self.res2(x)
                return x

        data = torch.rand((1, 3, 10, 10), dtype=torch.float)
        result = {True: 3, False: 1}
        for tracing in [True, False]:
            if tracing:
                m = torch.jit.trace(M(), data).eval()
            else:
                m = torch.jit.script(M()).eval()
            m = prepare_jit(m, {"": default_qconfig})
            if len(attrs_with_prefix(m, "_observer_")) != result[tracing]:
                raise AssertionError(
                    f"Expected {result[tracing]} observers, "
                    f"got {len(attrs_with_prefix(m, '_observer_'))}"
                )

    def test_insert_observers_for_if_consistent_observation(self):
        # [FUNCTION DOC] Checks that quantization with conditionals works when branch outputs are
        # observed consistently. For module M (conv then optional flatten) every combination of
        # cond and tracing must give 2 root observers; for module M2 (conv1 then either conv2 or
        # flatten) 3 observers are expected except 2 when traced with cond false.
        """check quantization for if works as long as
        output of all branches are quantized/observed consistently
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by an optional flatten selected by the
            # cond attribute.
            def __init__(self, cond):
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 3) and stores cond.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.cond = cond

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, then flattens the result if cond is true,
                # and returns it.
                x = self.conv(x)
                # x is already observed
                if self.cond:
                    x = torch.flatten(x)
                return x

        class M2(torch.nn.Module):
            # [CLASS DOC] HELPER: conv1 followed by a branch applying conv2 when cond is true or
            # flatten otherwise.
            def __init__(self, cond):
                # [FUNCTION DOC] HELPER: Creates two float convs (3 to 3, kernel 3) and
                # stores cond.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()
                self.cond = cond

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv1 then conv2 or flatten depending on cond;
                # returns the result.
                x = self.conv1(x)
                if self.cond:
                    x = self.conv2(x)
                    # x will be observed in the branch
                else:
                    x = torch.flatten(x)
                # since output for both branch are quantized
                # the if node is quantized consistently
                return x

        data = torch.rand((1, 3, 5, 5), dtype=torch.float)
        options = list(itertools.product([True, False], [True, False]))
        for cond, tracing in options:
            if tracing:
                m = torch.jit.trace(M(cond), data)
            else:
                m = torch.jit.script(M(cond))
            m = prepare_jit(m, {"": default_qconfig})
            if len(attrs_with_prefix(m, "_observer_")) != 2:
                raise AssertionError(
                    f"Expected 2 observers, got {len(attrs_with_prefix(m, '_observer_'))}"
                )

        for cond, tracing in options:
            if tracing:
                m = torch.jit.trace(M2(cond), data)
            else:
                m = torch.jit.script(M2(cond))
            m = prepare_jit(m, {"": default_qconfig})
            num_observers = 2 if tracing and not cond else 3
            if len(attrs_with_prefix(m, "_observer_")) != num_observers:
                raise AssertionError(
                    f"Expected {num_observers} observers, "
                    f"got {len(attrs_with_prefix(m, '_observer_'))}"
                )

    def test_insert_quant_dequant(self):
        # [FUNCTION DOC] Checks insert quant/dequant through convert_jit with debug on. For
        # per-channel and per-tensor weight observers, prepares a single-conv script module,
        # calibrates once, converts, requires a single submodule (conv), runs the quantized model,
        # and requires exactly 3 quantize nodes of the matching kind in the root graph.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one float conv (3 to 5, kernel 3).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv applied to x.
                return self.conv(x)

        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            observer = (
                default_per_channel_weight_observer.with_args(ch_axis=1)
                if is_per_channel
                else default_observer
            )
            qconfig_dict = {"": QConfig(activation=observer, weight=observer)}
            m = prepare_jit(m, qconfig_dict)
            data = torch.randn(1, 3, 10, 10, dtype=torch.float)

            m(data)
            m = convert_jit(m, debug=True)
            if len(m._modules._c.items()) != 1:
                raise AssertionError("Expected to have single submodule of conv")
            # make sure the quantized model is executable
            m(data)
            quant_func = (
                "aten::quantize_per_channel"
                if is_per_channel
                else "aten::quantize_per_tensor"
            )
            FileCheck().check_count(quant_func, 3, exactly=True).run(m.graph)

    def test_insert_quant_dequant_shared_class_type(self):
        # [FUNCTION DOC] Checks quant/dequant insertion for two convs sharing one class type, for
        # per-channel and per-tensor weight observers. Requires 3 root observers and 1 on each conv
        # after prepare, runs convert_jit with debug, requires the two convs still share a type and
        # all observers are gone, and uses FileCheck on each conv forward and on its _conv_forward
        # method for quantize and dequantize placement.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two float convs (3 to 3, kernel 3) applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv1 and conv2.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv2 of conv1 of x.
                return self.conv2(self.conv1(x))

        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            observer = (
                default_per_channel_weight_observer.with_args(ch_axis=1)
                if is_per_channel
                else default_observer
            )
            qconfig = QConfig(activation=observer, weight=observer)
            qconfig_dict = {"": qconfig}
            m = prepare_jit(m, qconfig_dict)
            # observers for input, output and value between conv1/conv2
            if len(attrs_with_prefix(m, "_observer_")) != 3:
                raise AssertionError("Expected to have 3 observers")
            # observer for weight
            if len(attrs_with_prefix(m.conv1, "_observer_")) != 1:
                raise AssertionError("Expected to have 1 observers")
            # observer for weight
            if len(attrs_with_prefix(m.conv2, "_observer_")) != 1:
                raise AssertionError("Expected to have 1 observers")

            data = torch.randn(1, 3, 10, 10, dtype=torch.float)
            m(data)
            m = convert_jit(m, debug=True)
            m(data)
            if m.conv1._c._type() != m.conv2._c._type():
                raise AssertionError(
                    f"Expected conv1 and conv2 to share the same type, "
                    f"got {m.conv1._c._type()} and {m.conv2._c._type()}"
                )

            # check all observers have been removed
            if len(attrs_with_prefix(m, "_observer_")) != 0:
                raise AssertionError("Expected to have 0 observers")
            if len(attrs_with_prefix(m.conv1, "_observer_")) != 0:
                raise AssertionError("Expected to have 0 observers")
            if len(attrs_with_prefix(m.conv2, "_observer_")) != 0:
                raise AssertionError("Expected to have 0 observers")

            quant_func = (
                "aten::quantize_per_channel"
                if is_per_channel
                else "aten::quantize_per_tensor"
            )
            for module in ["conv1", "conv2"]:
                conv = m._c.getattr(module)
                # quantize weight
                FileCheck().check(quant_func).check_next("aten::dequantize").check(
                    'prim::CallMethod[name="_conv_forward"]'
                ).check("return").run(get_forward_graph(conv))
                # no quantize node in _conv_forward
                FileCheck().check_not(quant_func).check("aten::conv2d").check_not(
                    quant_func
                ).check("return").run(conv._get_method("_conv_forward").graph)

    def test_dedup_module_uses(self):
        # [FUNCTION DOC] Checks the pass that dedups multiple uses of one submodule. Scripts a
        # module calling the same ReLU twice, requires one relu submodule before the pass and two
        # after torch._C._jit_pass_dedup_module_uses, and that outputs match.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that calls a single ReLU submodule twice with an in-place
            # subtract of 0.5 in between.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the relu module.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies relu, subtracts 0.5 in place, applies relu again
                # and returns it.
                x = self.relu(x)
                x -= 0.5
                return self.relu(x)

        data = torch.randn((2, 2))
        m = torch.jit.script(M())
        ref_res = m(data)
        if len([x for x, _ in m._modules._c.items() if x.startswith("relu")]) != 1:
            raise AssertionError(
                "Expected to have 1 relu modules after dedup module uses"
            )
        torch._C._jit_pass_dedup_module_uses(m._c)
        m = torch.jit._recursive.wrap_cpp_module(m._c)
        res = m(data)
        if len([x for x, _ in m._modules._c.items() if x.startswith("relu")]) != 2:
            raise AssertionError(
                "Expected to have 2 relu modules after dedup module uses"
            )
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize(self):
        # [FUNCTION DOC] Checks the replicate-dequantize pass. A quantized input is dequantized
        # once and used twice; before the pass the graph has one dequantize, after it two, and the
        # output equals the pre-pass output.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Dequantizes its input, applies a conv, and adds the dequantized
            # input in place to the conv output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 1).
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Dequantizes x, computes conv of it, adds the dequantized x
                # in place and returns the sum.
                x = torch.dequantize(x)
                r = self.conv(x)
                r += x
                return r

        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M())
        ref_res = m(x)
        FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)
        torch._C._jit_pass_replicate_dequantize(m.graph)
        FileCheck().check_count("aten::dequantize", 2, exactly=True).run(m.graph)
        res = get_forward(m._c)(x)
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize_in_block(self):
        # [FUNCTION DOC] Checks replicate-dequantize when uses are inside branches. Before the pass
        # there is one dequantize, afterwards two; FileCheck verifies a dequantize is right before
        # the conv call and right before the add; output equality is asserted.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Dequantizes the input, then applies conv or adds 3 depending
            # on cond.
            def __init__(self, cond):
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 1) and stores cond.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

                self.cond = cond

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Dequantizes x; returns conv of it when cond is true,
                # otherwise x plus 3.
                x = torch.dequantize(x)
                if self.cond:
                    x = self.conv(x)
                else:
                    x = x + 3
                return x

        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M(True))
        ref_res = m(x)
        FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)
        torch._C._jit_pass_replicate_dequantize(m.graph)
        FileCheck().check_count("aten::dequantize", 2, exactly=True).run(m.graph)
        # check dequantize is right before CallMethod of conv
        FileCheck().check("aten::dequantize").check_next("CallMethod").run(m.graph)
        # check dequantize is right before add
        FileCheck().check("aten::dequantize").check("aten::dequantize").check_next(
            "aten::add"
        ).run(m.graph)
        res = get_forward(m._c)(x)
        self.assertEqual(res, ref_res)

    def test_swap_functional_linear(self):
        # [FUNCTION DOC] Checks the pass that swaps functions named linear to aten::linear. Before
        # the pass the graph has a function call; after torch._C._jit_pass_swap_functional_linear
        # it has aten::linear and no function call; outputs equal. The comment notes the pass is no
        # longer needed.
        # TODO: This pass replaces any function called "linear" with "aten::linear"
        # No longer necessary, and also quite surprising
        def linear(input, weight, bias):
            # [FUNCTION DOC] HELPER: Local function named linear that forwards to the functional
            # linear op with input, weight and bias.
            return torch.nn.functional.linear(input, weight, bias)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Dequantizes the input and weight, calls the local linear
            # function, requantizes with scale 1.0, zero point 0 and quint8, and returns that.
            def forward(self, x, weight, bias):
                # [FUNCTION DOC] HELPER: Dequantizes x and weight, applies the local linear with
                # bias, quantizes per tensor and returns it.
                x = torch.dequantize(x)
                weight = torch.dequantize(weight)
                x = linear(x, weight, bias)
                x = torch.quantize_per_tensor(
                    x, scale=1.0, zero_point=0, dtype=torch.quint8
                )
                return x

        x = torch.rand((10, 5), dtype=torch.float)
        x = torch.quantize_per_tensor(x, scale=0.5, zero_point=1, dtype=torch.quint8)
        weight = torch.rand((5, 5), dtype=torch.float)
        weight = torch.quantize_per_tensor(
            weight, scale=0.5, zero_point=1, dtype=torch.qint8
        )
        bias = torch.rand((5), dtype=torch.float)
        m = torch.jit.script(M())
        ref_res = m(x, weight, bias)
        FileCheck().check("CallFunction").run(m.graph)
        torch._C._jit_pass_swap_functional_linear(m.graph)
        FileCheck().check("aten::linear").check_not("CallFunction").run(m.graph)
        res = m(x, weight, bias)
        self.assertEqual(res, ref_res)

    def test_replicate_quantize_for_if(self):
        # [FUNCTION DOC] Checks that quantize nodes after an if-node output are moved into the
        # branches so patterns fuse. Scripts two Res blocks, runs quantize_jit with calibration
        # data, and requires exactly 4 quantized conv calls in the graph.
        """We want to move quantize nodes for output of prim::If
        inside the prim::If blocks so that we can match quantization
        patterns.
        """

        class Res(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two convs choosing one by a boolean argument; stores
            # the flag as a mutable attribute so it is not frozen.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv and conv2 (3 to 3, kernel 1) and use_skip
                # initialised to true.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = True

            def forward(self, x: torch.Tensor, cond: bool) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Assigns cond to the use_skip attribute, then returns conv
                # of x if true, else conv2 of x.
                # to avoid being frozen
                self.use_skip = cond
                if self.use_skip:
                    return self.conv(x)
                else:
                    return self.conv2(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two Res blocks; res1 is called with true, res2 with false.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates res1 and res2.
                super().__init__()
                self.res1 = Res()
                self.res2 = Res()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies res1 with true then res2 with false and returns
                # the result.
                x = self.res1(x, True)
                x = self.res2(x, False)
                return x

        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        m = torch.jit.script(M()).eval()
        m = quantize_jit(m, qconfig_dict, test_only_eval_fn, [data])
        # make sure patterns in both branches are fused
        FileCheck().check_count("quantized::conv2d(", 4, exactly=True).run(m.graph)

    def test_finalize_for_linear(self):
        # [FUNCTION DOC] Checks finalization for a linear layer: after quantize_jit there is
        # exactly one per-tensor quantize node, no prepack node (it is folded) and a quantized
        # linear op.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one float Linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc applied to x.
                return self.fc(x)

        data = [[torch.rand((1, 5), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data])
        # make sure there is only one quantize_per_tensor for input
        # and linear_prepack is folded
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).check_not(
            "quantized::linear_prepack"
        ).check("quantized::linear").run(model.graph)

    def test_inplace_option(self):
        # [FUNCTION DOC] Checks the inplace option of quantize_jit. For traced and scripted conv,
        # calls quantize_jit with inplace on and ignores the return value; then requires the
        # original model graph to contain a quantized conv and no float conv.
        for tracing in [True, False]:
            model = get_script_module(
                torch.nn.Conv2d(3, 3, 3).float(), tracing, self.img_data_2d[0][0]
            )
            qconfig_dict = {"": default_qconfig}
            quantize_jit(
                model, qconfig_dict, test_only_eval_fn, [self.img_data_2d], inplace=True
            )
            FileCheck().check("quantized::conv2d").run(model.graph)

            FileCheck().check_not("aten::conv2d").run(model.graph)

    def test_finalize_debug(self):
        # [FUNCTION DOC] Checks quantize_jit with debug on keeps float ops with quantize/dequantize
        # around them. Quantizes conv plus avgpool; FileCheck requires no quantized conv, the float
        # conv and avg pool still present, and q_scale, q_zero_point and dtype queries following.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by average pooling (kernel 3).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 3) and AvgPool2d
                # (kernel 3).
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AvgPool2d(3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then avgpool and returns the result.
                x = self.conv(x)
                x = self.avgpool(x)
                return x

        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data], debug=True)
        FileCheck().check_not("quantized::conv2d").check("aten::conv2d").check(
            "aten::avg_pool2d"
        ).check("aten::q_scale").check_next("aten::q_zero_point").check_next(
            "prim::dtype"
        ).check_next("aten::quantize_per_tensor").check("aten::dequantize").run(
            model.graph
        )

    def test_module_list(self):
        # [FUNCTION DOC] Checks static quantization of linears held in a ModuleList and iterated in
        # forward. Scripts the model, requires 3 observer attributes on the root after prepare_jit,
        # calibrates once, converts without debug, and uses FileCheck to require two quantized
        # linear ops in the graph.
        class SimpleLinearLayer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping one float Linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc applied to x.
                return self.fc(x)

        class ComplexModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a ModuleList of two SimpleLinearLayer instances; forward
            # applies every layer to the same input and returns the list of outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ModuleList layers with two
                # SimpleLinearLayer entries.
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [SimpleLinearLayer() for i in range(2)]
                )

            def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Loops over layers, applies each to x, collects the outputs
                # into a list and returns it.
                states = []
                for layer in self.layers:
                    val = layer(x)
                    states.append(val)
                return states

        data = torch.rand((1, 5), dtype=torch.float)
        qconfig_dict = {"": default_qconfig}
        model = torch.jit.script(ComplexModel()).eval()
        model = prepare_jit(model, qconfig_dict)
        if len(attrs_with_prefix(model, "_observer")) != 3:
            raise AssertionError(
                f"Expected 3 observers, got {len(attrs_with_prefix(model, '_observer'))}"
            )
        model(data)
        model = convert_jit(model, debug=False)
        FileCheck().check("quantized::linear").check("quantized::linear").run(
            model.graph
        )

    def test_conv_trace(self):
        # [FUNCTION DOC] Checks that tracing and prepare_jit leave 1d, 2d and 3d convolutions as
        # the matching conv ops rather than the generic convolution op. Traces a module with the
        # three convs, runs prepare_jit, and uses FileCheck on each conv forward graph to require
        # the specific conv op and no generic convolution.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with Conv1d, Conv2d and Conv3d (3 to 3, kernel 3), each
            # applied to its own input; returns the three outputs as a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv1d, conv2d and conv3d float convs.
                super().__init__()
                self.conv1d = torch.nn.Conv1d(3, 3, 3).float()
                self.conv2d = torch.nn.Conv2d(3, 3, 3).float()
                self.conv3d = torch.nn.Conv3d(3, 3, 3).float()

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Applies conv1d to x, conv2d to y, conv3d to z and returns
                # the tuple of the three results.
                a = self.conv1d(x)
                b = self.conv2d(y)
                c = self.conv3d(z)
                return (a, b, c)

        qconfig_dict = {"": default_qconfig}
        inputs = (
            torch.rand((1, 3, 10), dtype=torch.float),
            torch.rand((1, 3, 10, 10), dtype=torch.float),
            torch.rand((1, 3, 10, 10, 10), dtype=torch.float),
        )
        model = torch.jit.trace(M(), inputs).eval()
        m = prepare_jit(model, qconfig_dict)
        FileCheck().check("aten::conv1d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.conv1d._c))
        )
        FileCheck().check("aten::conv2d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.conv2d._c))
        )
        FileCheck().check("aten::conv3d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.conv3d._c))
        )

    def test_convtranspose_trace(self):
        # [FUNCTION DOC] Same as the conv trace test but for transposed convolutions: after trace
        # and prepare_jit, each of the three submodule graphs must contain the specific transposed
        # conv op and no generic convolution op.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with ConvTranspose1d, 2d and 3d (3 to 3, kernel 3), each
            # applied to its own input; returns a tuple of the three outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates convtranspose1d, convtranspose2d and
                # convtranspose3d float layers.
                super().__init__()
                self.convtranspose1d = torch.nn.ConvTranspose1d(3, 3, 3).float()
                self.convtranspose2d = torch.nn.ConvTranspose2d(3, 3, 3).float()
                self.convtranspose3d = torch.nn.ConvTranspose3d(3, 3, 3).float()

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Applies the 1d layer to x, 2d to y, 3d to z and returns
                # the tuple of the results.
                a = self.convtranspose1d(x)
                b = self.convtranspose2d(y)
                c = self.convtranspose3d(z)
                return (a, b, c)

        qconfig_dict = {"": default_qconfig}
        inputs = (
            torch.rand((1, 3, 10), dtype=torch.float),
            torch.rand((1, 3, 10, 10), dtype=torch.float),
            torch.rand((1, 3, 10, 10, 10), dtype=torch.float),
        )
        model = torch.jit.trace(M(), inputs).eval()
        m = prepare_jit(model, qconfig_dict)
        FileCheck().check("aten::conv_transpose1d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.convtranspose1d._c))
        )
        FileCheck().check("aten::conv_transpose2d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.convtranspose2d._c))
        )
        FileCheck().check("aten::conv_transpose3d").check_not("aten::_convolution").run(
            str(get_forward_graph(m.convtranspose3d._c))
        )

    @unittest.skipUnless(
        "fbgemm" in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_replicate_dequant_same_value(self):
        # [FUNCTION DOC] Checks that when a quantized value is used twice by one op (multiplied by
        # itself), the pipeline still produces a single quantized mul and no float mul. Quantizes
        # conv-then-square with quantize_jit and FileCheck on the graph. Only runs when the fbgemm
        # engine is supported.
        class Mul(torch.nn.Module):
            # [CLASS DOC] HELPER: Applies a conv and returns the elementwise product of the output
            # with itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (3 to 3, kernel 3).
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv(x) multiplied by itself.
                x = self.conv(x)
                return x * x

        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        model = torch.jit.script(Mul()).eval()
        m = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data])
        FileCheck().check("quantized::mul(").check_not("aten::mul").run(m.graph)

    def test_interface_with_fork(self):
        # [FUNCTION DOC] Checks embedding-bag quantization through a fork call and an
        # interface-typed attribute. Traces a module that forks the forward of a test module which
        # has an interface-typed attribute and an EmbeddingBag submodule, uses placeholder-observer
        # qconfig named for the byte embedding-bag op, runs prepare_jit and convert_jit, and
        # requires the rowwise-offsets quantized embedding-bag op in the graph. Index and offset
        # tensors are fixed constants.
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Wraps one sum-mode EmbeddingBag (10 embeddings, dim 12, include
            # last offset, dense).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the EmbeddingBag attribute embedding1.
                super().__init__()
                self.embedding1 = torch.nn.EmbeddingBag(
                    num_embeddings=10,
                    embedding_dim=12,
                    include_last_offset=True,
                    sparse=False,
                    mode="sum",
                )

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns embedding1 applied to indices x and offsets y.
                return self.embedding1(x, y)

        class OrigMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Same structure as SubModule: one sum-mode EmbeddingBag; used as
            # the concrete value of an interface-typed attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the EmbeddingBag attribute embedding1.
                super().__init__()
                self.embedding1 = torch.nn.EmbeddingBag(
                    num_embeddings=10,
                    embedding_dim=12,
                    include_last_offset=True,
                    sparse=False,
                    mode="sum",
                )

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns embedding1 applied to indices x and offsets y.
                return self.embedding1(x, y)

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: TorchScript interface for a module taking two tensors and
            # returning a tensor; the body only passes.
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface forward declaration with no implementation.
                pass

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed attribute proxy_mod (holding an
            # OrigMod) and a concrete sub (a SubModule); forward calls both and returns only the
            # sub result.
            proxy_mod: ModInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates proxy_mod as OrigMod and sub as SubModule.
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls proxy_mod and sub with x and y, discards the proxy
                # result and returns the sub result.
                a = self.proxy_mod(x, y)
                b = self.sub(x, y)
                return b

        class MainModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Top-level module that forks the forward method of a TestModule
            # and waits for the result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the TestModule attribute test.
                super().__init__()
                self.test = TestModule()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Forks test.forward with x and y, waits on the future and
                # returns the result.
                fut = torch.jit._fork(self.test.forward, x, y)
                z = torch.jit._wait(fut)
                return z

        indices = torch.tensor(
            [
                9,
                6,
                5,
                7,
                8,
                8,
                9,
                2,
                8,
                6,
                6,
                9,
                1,
                6,
                8,
                8,
                3,
                2,
                3,
                6,
                3,
                6,
                5,
                7,
                0,
                8,
                4,
                6,
                5,
                8,
                2,
                3,
            ]
        )
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        m = torch.jit.trace(MainModule(), (indices, offsets))
        m.eval()

        int8_qconfig = QConfig(
            activation=PlaceholderObserver.with_args(
                dtype=torch.float, custom_op_name="embedding_bag_byte"
            ),
            weight=PlaceholderObserver.with_args(custom_op_name="embedding_bag_byte"),
        )

        m = prepare_jit(m, {"": int8_qconfig})
        m = convert_jit(m)
        FileCheck().check("quantized::embedding_bag_byte_rowwise_offsets").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantize_fork_wait(self):
        # [FUNCTION DOC] Checks dynamic quantization when fork and wait calls live in different
        # subgraphs. Traces a main module that waits on the future returned by a fork module, runs
        # prepare_dynamic_jit and convert_dynamic_jit with the default qconfig, requires a dynamic
        # quantized linear op in the graph, and saves the model to an in-memory buffer to confirm
        # saving works. Skipped without fbgemm.
        """Tests the case where fork and wait calls are in different subgraphs
        Calling inline fork-wait only removes the fork call and leaves aten::wait
        calls in the graph, with Tensor as input (instead of Future[Tensor])
        """

        class MainModule(nn.Module):
            # [CLASS DOC] HELPER: Owns a ForkModule as fork_ops; forward waits on the future
            # returned by it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ForkModule attribute fork_ops.
                super().__init__()
                self.fork_ops = ForkModule()

            def init_values(self, x):
                # [FUNCTION DOC] HELPER: Calls fork_ops on x and stores the result as attribute
                # fork_dict; never called by the test.
                shared_module = self.fork_ops(x)
                self.fork_dict = shared_module

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Waits on the future returned by fork_ops(x) and returns
                # the value.
                val = torch.jit._wait(self.fork_ops(x))
                return val

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no state whose forward applies functional linear with
            # a ones weight (5 by 5) and zeros bias created inside forward.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds ones weight and zeros bias, returns functional
                # linear of x with them.
                w = torch.ones(5, 5)
                b = torch.zeros(5)
                return torch.nn.functional.linear(x, w, b)

        class ForkModule(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward forks the forward of a TestModule and
            # returns the future rather than the value.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the TestModule attribute test.
                super().__init__()
                self.test = TestModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forks test.forward on x and returns the future.
                fut = torch.jit._fork(self.test.forward, x)
                return fut

        model = MainModule().eval()
        traced = torch.jit.trace(model, (torch.randn(5, 5),))
        model = prepare_dynamic_jit(traced, {"": default_qconfig})
        model = convert_dynamic_jit(model)
        FileCheck().check("quantized::linear_dynamic").run(model.graph)
        # Make sure model save works
        b = io.BytesIO()
        torch.jit.save(model, b)


class TestQuantizeJitOps(QuantizationTestCase):
    # [CLASS DOC] Tests that graph-mode post training static quantization (quantize_jit through the
    # checkGraphModeOp helper of QuantizationTestCase) produces the expected quantized operator end
    # to end, for linear, conv, add, mul, cat, batch norm, activations and similar ops, in both
    # traced and scripted forms. Most tests are skipped without the fbgemm backend.
    """Test graph mode post training static quantization works
    for individual ops end to end.
    """

    @skipIfNoFBGEMM
    def test_linear(self):
        # [FUNCTION DOC] Checks linear quantization. Without relu, both a Linear module and a
        # functional-linear module, traced and scripted, must give a quantized linear op, exactly
        # one input quantize node and no prepack op. With relu (module relu, functional relu or
        # identity-free combos) the fused quantized linear-relu op must appear and no separate
        # linear, relu or plain quantized linear/relu op.
        class ModuleLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Float Linear (30 to 4) followed by an activation chosen at
            # construction: functional relu, a ReLU module, or Identity when relu is not wanted.
            def __init__(self, has_relu=False, f_relu=False):
                # [FUNCTION DOC] HELPER: Creates linear and sets attribute relu to the functional
                # relu, a ReLU module, or an Identity based on has_relu and f_relu.
                super().__init__()
                self.linear = torch.nn.Linear(30, 4).float()
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu applied to the linear output.
                return self.relu(self.linear(x))

        class FuncLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Functional linear with random plain-tensor weight (4 by 30) and
            # bias (4) followed by an activation chosen as in ModuleLinear.
            def __init__(self, has_relu=False, f_relu=False):
                # [FUNCTION DOC] HELPER: Creates random weight and bias tensors and sets attribute
                # relu as in ModuleLinear.
                super().__init__()
                self.w = torch.randn(4, 30)
                self.b = torch.randn(4)
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu applied to functional linear of x with the
                # stored weight and bias.
                return self.relu(F.linear(x, self.w, self.b))

        data = [[torch.rand((1, 30), dtype=torch.float)]]
        for model, tracing in itertools.product(
            [ModuleLinear(has_relu=False), FuncLinear(has_relu=False)], [True, False]
        ):
            model = self.checkGraphModeOp(model, data, "quantized::linear", tracing)
            FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).run(
                model.graph
            )
            FileCheck().check_not("quantized::linear_prepack").run(model.graph)

        for f_relu, tracing in itertools.product([True, False], [True, False]):
            for model in [
                ModuleLinear(has_relu=True, f_relu=f_relu),
                FuncLinear(has_relu=True, f_relu=f_relu),
            ]:
                model = self.checkGraphModeOp(
                    model, data, "quantized::linear_relu", tracing
                )
                checker = (
                    FileCheck()
                    .check_not("aten::linear")
                    .check_not("aten::relu")
                    .check_not("quantized::linear(")
                    .check_not("quantized::relu(")
                    .run(model.graph)
                )

    @skipIfNoFBGEMM
    def test_quantized_conv(self):
        # [FUNCTION DOC] Checks conv quantization for 1d, 2d and 3d, traced and scripted: the
        # quantized conv op for that dimension appears, there is exactly one input quantize node
        # and the prepack op is folded away.
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class Conv(torch.nn.Module):
            # [CLASS DOC] HELPER: Single conv of the dimension selected from the enclosing mapping
            # (3 to 3, kernel 3).
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates float conv of dimension dim as conv.
                super().__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv applied to x.
                return self.conv(x)

        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            model = self.checkGraphModeOp(
                Conv(dim),
                self.img_data_dict[dim],
                f"quantized::conv{dim}d",
                tracing,
            )
            # make sure there is only one quantize_per_tensor for input
            # and conv2d_prepack is folded
            FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).run(
                model.graph
            )

            FileCheck().check_not(f"quantized::conv{dim}d_prepack").run(model.graph)

    @skipIfNoFBGEMM
    def test_quantized_conv_relu(self):
        # [FUNCTION DOC] Checks fused conv-relu quantization for 1d, 2d and 3d with four relu
        # variants (in-place and out-of-place module, functional, in-place functional), traced and
        # scripted. Requires the quantized conv-relu op and no float conv, relu, quantized conv or
        # quantized relu.
        """tests for conv1d_relu/conv2d_relu/conv3d_relu"""
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class ConvNdRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by a ReLU module whose in-place flag is a
            # constructor argument.
            def __init__(self, dim, inplace):
                # [FUNCTION DOC] HELPER: Creates the conv for dimension dim and ReLU with the given
                # inplace flag.
                super().__init__()
                self.conv = conv_module[dim](3, 3, 3).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu module applied to the conv output.
                return self.relu(self.conv(x))

        class ConvNdFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by out-of-place functional relu.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates the conv for dimension dim.
                super().__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional relu of the conv output.
                return F.relu(self.conv(x))

        class ConvNdInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by in-place functional relu.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates the conv for dimension dim.
                super().__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional relu applied in place to the
                # conv output.
                return F.relu(self.conv(x), True)

        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            for orig_m in [
                ConvNdRelu(dim, True),
                ConvNdRelu(dim, False),
                ConvNdFunctionalRelu(dim),
                ConvNdInplaceFunctionalRelu(dim),
            ]:
                conv_name = f"conv{dim}d"
                m = self.checkGraphModeOp(
                    orig_m,
                    self.img_data_dict[dim],
                    f"quantized::conv{dim}d_relu(",
                    tracing=tracing,
                )

                FileCheck().check_not(f"aten::conv{dim}d(").check_not(
                    "aten::relu"
                ).check_not(f"quantized::conv{dim}d(").check_not(
                    "quantized::relu("
                ).run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_alpha(self):
        # [FUNCTION DOC] Checks that three add ops sharing the same constant alpha argument are all
        # converted: exactly 3 quantized add ops and no float add or in-place add remain, for
        # traced and scripted.
        """Test quant fusion for multiple aten::add using same
        constant alpha as the third argument
        """

        class QuantizedAdd(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs on separate inputs followed by three chained adds (x
            # plus y, y plus z, z plus w).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies conv1 to x and conv2 to y, then computes z as
                # their sum, w as y plus z, and returns z plus w.
                x = self.conv1(x)
                y = self.conv2(y)
                z = x + y
                w = y + z
                return z + w

        data = [
            [
                torch.randn(1, 2, 5, 5, dtype=torch.float),
                torch.randn(1, 2, 5, 5, dtype=torch.float),
            ]
        ]
        for tracing in [True, False]:
            m = self.checkGraphModeOp(QuantizedAdd(), data, "quantized::add", tracing)
            FileCheck().check_count("quantized::add", 3, exactly=True).run(m.graph)
            FileCheck().check_not("aten::add").check_not("aten::add_").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_relu_alpha(self):
        # [FUNCTION DOC] Checks fusion of add followed by relu for eight module variants (module
        # relu in or out of place, plain or in-place add, functional relu variants) in traced and
        # scripted form: each graph must contain exactly 2 quantized add-relu ops and none of the
        # float or non-fused ops.
        """Test quant fusion for multiple aten::add using same
        constant alpha as the third argument in add_relu pattern
        """

        class AddRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, add, relu module, add again, relu module; the relu
            # module in-place flag is a constructor argument.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2) and a ReLU with
                # the given inplace flag.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, then add, relu, add y again and relu,
                # returning the result (uses out-of-place adds).
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = self.relu(x)
                x = x + y
                return self.relu(x)

        class InplaceAddRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Same as AddRelu but uses in-place add operations.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2) and a ReLU with
                # the given inplace flag.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, then in-place add, relu, in-place add
                # of y, relu and returns the result.
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = self.relu(x)
                x += y
                return self.relu(x)

        class AddFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Out-of-place add pattern with out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, then add, functional relu, add y,
                # functional relu, and returns the result.
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = F.relu(x)
                x = x + y
                return F.relu(x)

        class InplaceAddFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: In-place add pattern with out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, then in-place add, functional relu,
                # in-place add of y, functional relu, and returns the result.
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = F.relu(x)
                x += y
                return F.relu(x)

        class AddInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Out-of-place add with in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, then add, in-place functional relu, add
                # y, in-place functional relu, and returns the result.
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = F.relu(x, True)
                x = x + y
                return F.relu(x, True)

        class InplaceAddInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: In-place add with in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, then in-place add, in-place functional
                # relu, in-place add of y, in-place functional relu, and returns the result.
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = F.relu(x, True)
                x += y
                return F.relu(x, True)

        data = [
            [
                torch.rand((1, 2, 5, 5), dtype=torch.float),
                torch.rand((1, 2, 5, 5), dtype=torch.float),
            ]
        ]
        for m_orig in [
            AddRelu(True),
            AddRelu(False),
            InplaceAddRelu(True),
            InplaceAddRelu(False),
            AddFunctionalRelu(),
            InplaceAddFunctionalRelu(),
            AddInplaceFunctionalRelu(),
            InplaceAddInplaceFunctionalRelu(),
        ]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(
                    m_orig, data, "quantized::add_relu(", tracing=tracing
                )
                FileCheck().check_count("quantized::add_relu(", 2, exactly=True).run(
                    m.graph
                )
                FileCheck().check_not("aten::add(").check_not("aten::add_(").check_not(
                    "aten::relu("
                ).check_not("aten::relu_(").check_not("quantized::add(").check_not(
                    "quantized::relu("
                ).run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add(self):
        # [FUNCTION DOC] Checks add quantization. Conv-then-add modules (plain and in-place) must
        # produce the quantized add op with no float add; add modules without convs must stay float
        # adds with no quantized add. Done for traced and scripted.
        class QuantizedAdd(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs on separate inputs followed by an add.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs and returns the sum of the results.
                x = self.conv1(x)
                y = self.conv2(y)
                return x + y

        class QuantizedInplaceAdd(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs followed by an in-place add of the second result into
            # the first.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, adds y into x in place and returns x.
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return x

        class NonQuantizedAdd(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module that adds its two inputs; no conv so it is not
            # expected to be quantized.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return x + y

        class NonQuantizedInplaceAdd(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module that adds y into x in place and returns x; not
            # expected to be quantized.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Adds y into x in place and returns x.
                x += y
                return x

        data = [
            [
                torch.randn(1, 2, 3, 3, dtype=torch.float),
                torch.randn(1, 2, 3, 3, dtype=torch.float),
            ]
        ]
        for m, quantized in [
            (QuantizedAdd(), True),
            (QuantizedInplaceAdd(), True),
            (NonQuantizedAdd(), False),
            (NonQuantizedInplaceAdd(), False),
        ]:
            for tracing in [True, False]:
                op = "quantized::add" if quantized else "aten::add"
                m = self.checkGraphModeOp(m, data, op, tracing)
                # TODO: remove after refactor of checkGraphModeOp
                if quantized:
                    FileCheck().check_not("aten::add").check_not("aten::add_").run(
                        m.graph
                    )
                else:
                    FileCheck().check_not("quantized::add").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_scalar(self):
        # [FUNCTION DOC] Checks add-with-scalar quantization. Conv-then-add-3 modules (plain and
        # in-place) must give the quantized add-scalar op and no float add; bare add-3 modules
        # must stay float. Numerics are not compared because add-scalar is unsupported for the
        # numeric check.
        class QuantizedAddScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by adding the constant 3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv(x) plus 3.
                x = self.conv(x)
                return x + 3

        class QuantizedInplaceAddScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by an in-place add of the constant 3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, adds 3 in place, returns the result.
                x = self.conv(x)
                x += 3
                return x

        class NonQuantizedAddScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module returning x plus 3; not expected to
            # be quantized.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus 3.
                return x + 3

        class NonQuantizedInplaceAddScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module adding 3 to x in place; not expected to
            # be quantized.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 3 to x in place and returns x.
                x += 3
                return x

        data = [[torch.randn(1, 2, 3, 3, dtype=torch.float)]]
        for m, quantized in [
            (QuantizedAddScalar(), True),
            (QuantizedInplaceAddScalar(), True),
            (NonQuantizedAddScalar(), False),
            (NonQuantizedInplaceAddScalar(), False),
        ]:
            for tracing in [True, False]:
                op = "quantized::add_scalar" if quantized else "aten::add"
                # we don't check the numerical consistency for add_scalar
                # since it's not supported
                m = self.checkGraphModeOp(m, data, op, tracing, check=False)
                # TODO: remove after refactor of checkGraphModeOp
                if quantized:
                    FileCheck().check_not("aten::add").check_not("aten::add_").run(
                        m.graph
                    )
                else:
                    FileCheck().check_not("quantized::add_scalar").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_relu(self):
        # [FUNCTION DOC] Checks add followed by relu fusion across eight variants (relu module in
        # or out of place, plain or in-place add, functional relu plain or in place), each traced
        # and scripted. Each graph must contain the quantized add-relu op and none of the float
        # add, relu, quantized add or quantized relu ops.
        class AddRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, add, then ReLU module whose in-place flag is a
            # constructor argument.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2) and ReLU with
                # the given inplace flag.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, adds the results, applies relu module
                # and returns it.
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return self.relu(x)

        class InplaceAddRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, in-place add of the second result into the first, then
            # a ReLU module.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2) and a ReLU with
                # the given inplace flag.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, adds y into x in place, applies the
                # relu module and returns it.
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return self.relu(x)

        class AddFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, out-of-place add, then out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, adds, and returns functional relu of
                # the sum.
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return F.relu(x)

        class InplaceAddFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, in-place add, then out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, adds y into x in place and returns
                # functional relu of x.
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return F.relu(x)

        class AddInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, out-of-place add, then in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, adds, and returns in-place functional
                # relu of the sum.
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return F.relu(x, True)

        class InplaceAddInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, in-place add, then in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, adds y into x in place and returns
                # in-place functional relu of x.
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return F.relu(x, True)

        data = [
            [
                torch.rand((1, 2, 5, 5), dtype=torch.float),
                torch.rand((1, 2, 5, 5), dtype=torch.float),
            ]
        ]
        for m in [
            AddRelu(True),
            AddRelu(False),
            InplaceAddRelu(True),
            InplaceAddRelu(False),
            AddFunctionalRelu(),
            InplaceAddFunctionalRelu(),
            AddInplaceFunctionalRelu(),
            InplaceAddInplaceFunctionalRelu(),
        ]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m, data, "quantized::add_relu(", tracing)
                FileCheck().check_not("aten::add(").check_not("aten::add_(").check_not(
                    "aten::relu("
                ).check_not("aten::relu_(").check_not("quantized::add(").check_not(
                    "quantized::relu("
                ).run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_scalar_relu(self):
        # [FUNCTION DOC] Checks conv plus scalar add plus relu fusion across eight variants (module
        # relu in or out of place, plain or in-place add, functional relu plain or in place),
        # traced and scripted. Numeric comparison is disabled; the graph must have none of the
        # float add, relu, quantized add-scalar or quantized relu ops.
        class AddScalarRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, add of constant 3, then ReLU module with a constructor
            # in-place flag.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2) and ReLU with the
                # given inplace flag.
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv and returns relu module of the result plus 3.
                x = self.conv(x)
                return self.relu(x + 3)

        class InplaceAddScalarRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, in-place add of constant 3, then ReLU module.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2) and ReLU with the
                # given inplace flag.
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, adds 3 in place and returns relu module of
                # the result.
                x = self.conv(x)
                x += 3
                return self.relu(x)

        class AddScalarFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, add of constant 3, then out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv and returns functional relu of the result
                # plus 3.
                x = self.conv(x)
                return F.relu(x + 3)

        class InplaceAddScalarFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, in-place add of constant 3, then out-of-place
            # functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, adds 3 in place and returns functional relu
                # of the result.
                x = self.conv(x)
                x += 3
                return F.relu(x)

        class AddScalarInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, add of constant 3, then in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv and returns in-place functional relu of the
                # result plus 3.
                x = self.conv(x)
                return F.relu(x + 3, True)

        class InplaceAddScalarInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, in-place add of constant 3, then in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, adds 3 in place and returns in-place
                # functional relu of the result.
                x = self.conv(x)
                x += 3
                return F.relu(x, True)

        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float)]]
        for m in [
            AddScalarRelu(True),
            AddScalarRelu(False),
            InplaceAddScalarRelu(True),
            InplaceAddScalarRelu(False),
            AddScalarFunctionalRelu(),
            InplaceAddScalarFunctionalRelu(),
            AddScalarInplaceFunctionalRelu(),
            InplaceAddScalarInplaceFunctionalRelu(),
        ]:
            for tracing in [True, False]:
                # quantized::add_scalar_relu or quantized::add_scalar_relu_out
                # TODO: split this after refactor of checkGraphModeOp
                m = self.checkGraphModeOp(
                    m, data, "quantized::add_scalar_relu", tracing, check=False
                )
                FileCheck().check_not("aten::add(").check_not("aten::add_(").check_not(
                    "aten::relu("
                ).check_not("aten::relu_(").check_not(
                    "quantized::add_scalar("
                ).check_not("quantized::relu(").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_cat(self):
        # [FUNCTION DOC] Checks cat quantization depends on its inputs. Concatenation of two conv
        # outputs must become the quantized cat op with no float cat; concatenation of raw inputs
        # must stay a float cat with no quantized cat. Traced and scripted.
        """quantization of the output of cat will be depend on the
        input of cat. we only quantize the output of cat when its inputs are quantized.
        """

        class QuantizedCat(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs on separate inputs whose outputs are concatenated along
            # dimension 1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs and returns the concatenation along
                # dimension 1.
                x = self.conv1(x)
                y = self.conv2(y)
                return torch.cat([x, y], 1)

        class NonQuantizedCat(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module concatenating its two inputs along dimension 1;
            # not expected to be quantized.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the concatenation of x and y along dimension 1.
                return torch.cat([x, y], 1)

        data = [
            [
                torch.randn(1, 2, 5, 5, dtype=torch.float),
                torch.randn(1, 2, 5, 5, dtype=torch.float),
            ]
        ]
        for tracing in [True, False]:
            m = self.checkGraphModeOp(QuantizedCat(), data, "quantized::cat", tracing)
            FileCheck().check_not("aten::cat").run(m.graph)

            m = self.checkGraphModeOp(NonQuantizedCat(), data, "aten::cat", tracing)
            FileCheck().check_not("quantized::cat").run(m.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm(self):
        # [FUNCTION DOC] Checks batch norm quantization for 1d, 2d and 3d, traced and scripted: the
        # quantized batch norm op must appear and no float batch norm remain.
        bn_module = {
            1: torch.nn.BatchNorm1d,
            2: torch.nn.BatchNorm2d,
            3: torch.nn.BatchNorm3d,
        }

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Single float batch norm (3 features) of the
            # selected dimensionality.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates the batch norm of dimension dim as bn, converted
                # to float.
                super().__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns bn applied to x.
                return self.bn(x)

        options = itertools.product([True, False], [1, 2, 3])
        for tracing, dim in options:
            model = self.checkGraphModeOp(
                M(dim), self.img_data_dict[dim], "quantized::batch_norm", tracing
            )

            FileCheck().check_not("aten::batch_norm").run(model.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu_BNRelu(self):
        # [FUNCTION DOC] Checks batch norm plus ReLU module fusion for 2d and 3d with in-place and
        # out-of-place relu, traced and scripted: the quantized batch-norm-relu op must appear and
        # no float batch norm or relu remain.
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        class BNRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Batch norm followed by a ReLU module with a constructor
            # in-place flag.
            def __init__(self, dim, inplace):
                # [FUNCTION DOC] HELPER: Creates float batch norm (3 features) for dim and
                # the ReLU.
                super().__init__()
                self.bn = bn_module[dim](3).to(torch.float)
                self.relu = torch.nn.ReLU(inplace=inplace)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu module of the batch norm output.
                return self.relu(self.bn(x))

        options = itertools.product([True, False], [2, 3])
        for tracing, dim in options:
            for instance in [BNRelu(dim, True), BNRelu(dim, False)]:
                model = self.checkGraphModeOp(
                    instance,
                    self.img_data_dict[dim],
                    "quantized::batch_norm_relu",
                    tracing,
                )
                FileCheck().check_not("aten::batch_norm").check_not(
                    "aten::relu"
                ).check_not("aten::relu_").run(model.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu_BNFuncRelu(self):
        # [FUNCTION DOC] Checks batch norm plus out-of-place functional relu fusion for 2d and
        # 3d, traced and scripted; expects the quantized batch-norm-relu op and no float batch
        # norm or relu.
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        class BNFuncRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Batch norm followed by out-of-place functional relu.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates float batch norm (3 features) for dim.
                super().__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional relu (not in place) of the batch
                # norm output.
                return F.relu(self.bn(x), False)

        options = itertools.product([True, False], [2, 3])
        for tracing, dim in options:
            instance = BNFuncRelu(dim)
            model = self.checkGraphModeOp(
                instance, self.img_data_dict[dim], "quantized::batch_norm_relu", tracing
            )
            FileCheck().check_not("aten::batch_norm").check_not("aten::relu").check_not(
                "aten::relu_"
            ).run(model.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu_BNFuncInplaceRelu(self):
        # [FUNCTION DOC] Checks batch norm plus in-place functional relu fusion for 2d and 3d,
        # traced and scripted; expects the quantized batch-norm-relu op and no float batch norm
        # or relu.
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        class BNFuncInplaceRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Batch norm followed by in-place functional relu.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates float batch norm (3 features) for dim.
                super().__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns in-place functional relu of the batch norm output.
                return F.relu(self.bn(x), True)

        options = itertools.product([True, False], [2, 3])
        for tracing, dim in options:
            instance = BNFuncInplaceRelu(dim)
            model = self.checkGraphModeOp(
                instance, self.img_data_dict[dim], "quantized::batch_norm_relu", tracing
            )
            FileCheck().check_not("aten::batch_norm").check_not("aten::relu").check_not(
                "aten::relu_"
            ).run(model.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul(self):
        # [FUNCTION DOC] Checks mul quantization. Conv-then-mul modules (plain and in-place) must
        # give the quantized mul op and no float mul; bare mul modules must remain float with no
        # quantized mul. Traced and scripted.
        class QuantizedMul(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs on separate inputs, outputs multiplied.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs and returns the product.
                x = self.conv1(x)
                y = self.conv2(y)
                return x * y

        class QuantizedInplaceMul(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs followed by an in-place multiply of the first result by
            # the second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, multiplies x by y in place and
                # returns x.
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return x

        class NonQuantizedMul(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module returning the product of its inputs; not
            # expected to be quantized.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x times y.
                return x * y

        class NonQuantizedInplaceMul(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module multiplying x by y in place; not expected to
            # be quantized.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Multiplies x by y in place and returns x.
                x *= y
                return x

        data = [
            [
                torch.randn(1, 2, 10, 10, dtype=torch.float),
                torch.randn(1, 2, 10, 10, dtype=torch.float),
            ]
        ]
        for m, quantized in [
            (QuantizedMul(), True),
            (QuantizedInplaceMul(), True),
            (NonQuantizedMul(), False),
            (NonQuantizedInplaceMul(), False),
        ]:
            for tracing in [True, False]:
                op = "quantized::mul" if quantized else "aten::mul"
                m = self.checkGraphModeOp(m, data, op, tracing)
                # TODO: remove after refactor of checkGraphModeOp
                if quantized:
                    FileCheck().check_not("aten::mul").check_not("aten::mul_").run(
                        m.graph
                    )
                else:
                    FileCheck().check_not("quantized::mul").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_scalar(self):
        # [FUNCTION DOC] Checks mul-by-scalar quantization. Conv-then-multiply-by-3 modules (plain
        # and in-place) must give the quantized mul-scalar op and no float mul; bare multiply-by-3
        # modules must stay float. Numerics are not compared.
        class QuantizedMulScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by multiplication by constant 3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv(x) times 3.
                x = self.conv(x)
                return x * 3

        class QuantizedInplaceMulScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by in-place multiplication by constant 3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, multiplies by 3 in place and returns
                # the result.
                x = self.conv(x)
                x *= 3
                return x

        class NonQuantizedMulScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module returning x times 3; not expected to
            # be quantized.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x times 3.
                return x * 3

        class NonQuantizedInplaceMulScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module multiplying x by 3 in place; not expected to
            # be quantized.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies x by 3 in place and returns x.
                x *= 3
                return x

        data = [[torch.randn(1, 2, 5, 5, dtype=torch.float)]]
        for m, quantized in [
            (QuantizedMulScalar(), True),
            (QuantizedInplaceMulScalar(), True),
            (NonQuantizedMulScalar(), False),
            (NonQuantizedInplaceMulScalar(), False),
        ]:
            for tracing in [True, False]:
                op = "quantized::mul_scalar" if quantized else "aten::mul"
                # we don't check the numerical consistency for add_scalar
                # since it's not supported
                m = self.checkGraphModeOp(m, data, op, tracing, check=False)
                # TODO: remove after refactor of checkGraphModeOp
                if quantized:
                    FileCheck().check_not("aten::mul").check_not("aten::mul_").run(
                        m.graph
                    )
                else:
                    FileCheck().check_not("quantized::mul_scalar").run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_relu(self):
        # [FUNCTION DOC] Checks conv-conv mul followed by relu fusion across eight variants (relu
        # module in or out of place, plain or in-place mul, functional relu plain or in place),
        # traced and scripted: the quantized mul-relu op must appear and none of the float mul,
        # relu, quantized mul or quantized relu ops.
        class MulRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, mul, then ReLU module with a constructor
            # in-place flag.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2) and ReLU with
                # the given inplace flag.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, multiplies the results and returns relu
                # module of the product.
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return self.relu(x)

        class InplaceMulRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, in-place mul, then ReLU module.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2) and ReLU with
                # the given inplace flag.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, multiplies x by y in place and returns
                # relu module of x.
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return self.relu(x)

        class MulFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, out-of-place mul, then out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, multiplies and returns functional relu
                # of the product.
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return F.relu(x)

        class InplaceMulFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, in-place mul, then out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, multiplies x by y in place and returns
                # functional relu of x.
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return F.relu(x)

        class MulInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, out-of-place mul, then in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, multiplies and returns in-place
                # functional relu of the product.
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return F.relu(x, True)

        class InplaceMulInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Two convs, in-place mul, then in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two float convs (2 to 2, kernel 2).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the convs, multiplies x by y in place and returns
                # in-place functional relu of x.
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return F.relu(x, True)

        data = [
            [
                torch.rand((1, 2, 5, 5), dtype=torch.float),
                torch.rand((1, 2, 5, 5), dtype=torch.float),
            ]
        ]
        for m in [
            MulRelu(True),
            MulRelu(False),
            InplaceMulRelu(True),
            InplaceMulRelu(False),
            MulFunctionalRelu(),
            InplaceMulFunctionalRelu(),
            MulInplaceFunctionalRelu(),
            InplaceMulInplaceFunctionalRelu(),
        ]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m, data, "quantized::mul_relu(", tracing)
                FileCheck().check_not("aten::mul(").check_not("aten::mul_(").check_not(
                    "aten::relu("
                ).check_not("aten::relu_(").check_not("quantized::mul(").check_not(
                    "quantized::relu("
                ).run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_scalar_relu(self):
        # [FUNCTION DOC] Checks conv plus scalar mul plus relu fusion across eight variants, traced
        # and scripted. Numeric comparison is disabled; the graph must contain none of the float
        # mul, relu, quantized mul-scalar or quantized relu ops.
        class MulScalarRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, multiply by constant 3, then ReLU module with a constructor
            # in-place flag.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2) and ReLU with the
                # given inplace flag.
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv and returns relu module of the result
                # times 3.
                x = self.conv(x)
                return self.relu(x * 3)

        class InplaceMulScalarRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, in-place multiply by constant 3, then ReLU module.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2) and ReLU with the
                # given inplace flag.
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, multiplies by 3 in place and returns relu
                # module of the result.
                x = self.conv(x)
                x *= 3
                return self.relu(x)

        class MulScalarFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, multiply by constant 3, then out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv and returns functional relu of the result
                # times 3.
                x = self.conv(x)
                return F.relu(x * 3)

        class InplaceMulScalarFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, in-place multiply by 3, then out-of-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, multiplies by 3 in place and returns
                # functional relu of the result.
                x = self.conv(x)
                x *= 3
                return F.relu(x)

        class MulScalarInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, multiply by constant 3, then in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv and returns in-place functional relu of the
                # result times 3.
                x = self.conv(x)
                return F.relu(x * 3, True)

        class InplaceMulScalarInplaceFunctionalRelu(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv, in-place multiply by 3, then in-place functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, multiplies by 3 in place and returns
                # in-place functional relu of the result.
                x = self.conv(x)
                x *= 3
                return F.relu(x, True)

        data = [[torch.randn(1, 2, 5, 5, dtype=torch.float)]]
        for m in [
            MulScalarRelu(True),
            MulScalarRelu(False),
            InplaceMulScalarRelu(True),
            InplaceMulScalarRelu(False),
            MulScalarFunctionalRelu(),
            InplaceMulScalarFunctionalRelu(),
            MulScalarInplaceFunctionalRelu(),
            InplaceMulScalarInplaceFunctionalRelu(),
        ]:
            for tracing in [True, False]:
                # quantized::mul_scalar_relu or quantized::mul_scalar_relu_out
                m = self.checkGraphModeOp(
                    m, data, "quantized::mul_scalar_relu", tracing, check=False
                )
                FileCheck().check_not("aten::mul(").check_not("aten::mul_(").check_not(
                    "aten::relu("
                ).check_not("aten::relu_(").check_not(
                    "quantized::mul_scalar("
                ).check_not("quantized::relu(").run(m.graph)

    @override_qengines
    def test_hardswish(self):
        # [FUNCTION DOC] Checks hardswish quantization for the module form and the functional form
        # (in place and not), traced and scripted, under every available quantization engine
        # (override decorator). Requires the quantized hardswish op and no float hardswish or
        # in-place hardswish in the graph.
        class FunctionalHardswish(torch.nn.Module):
            # [CLASS DOC] HELPER: Applies functional hardswish with an in-place flag chosen
            # at construction.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Stores the inplace flag.
                super().__init__()
                self.inplace = inplace

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns functional hardswish of input using the stored
                # in-place flag.
                return torch.nn.functional.hardswish(input, inplace=self.inplace)

        modules = [
            torch.nn.Hardswish(),
            FunctionalHardswish(True),
            FunctionalHardswish(False),
        ]

        for test_case in itertools.product([True, False], modules):
            tracing, m = test_case
            m = self.checkGraphModeOp(
                m, self.img_data_2d, "quantized::hardswish", tracing
            )
            FileCheck().check_not("aten::hardswish").check_not("aten::hardswish_").run(
                m.graph
            )

    @override_qengines
    def test_elu(self):
        # [FUNCTION DOC] Checks ELU quantization for the module and functional forms with in-place
        # on and off, traced and scripted, under each quantization engine. Requires the quantized
        # elu op and no float elu or in-place elu.
        class FunctionalELU(torch.nn.Module):
            # [CLASS DOC] HELPER: Applies functional elu with an in-place flag chosen
            # at construction.
            def __init__(self, inplace=False):
                # [FUNCTION DOC] HELPER: Stores the inplace flag (default false).
                super().__init__()
                self.inplace = inplace

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns functional elu of input using the stored
                # in-place flag.
                return torch.nn.functional.elu(input, inplace=self.inplace)

        modules = [torch.nn.ELU, FunctionalELU]
        for test_case in itertools.product([True, False], [True, False], modules):
            tracing, inplace, mod_class = test_case
            m = mod_class(inplace=inplace)
            m = self.checkGraphModeOp(m, self.img_data_2d, "quantized::elu", tracing)
            FileCheck().check_not("aten::elu").check_not("aten::elu_").run(m.graph)

    @override_qengines
    def test_layer_norm(self):
        # [FUNCTION DOC] Checks layer norm quantization on a 2 by 5 by 5 normalized shape, traced
        # and scripted, under each quantization engine: the quantized layer norm op must appear and
        # no float layer norm remain.
        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float)] for _ in range(2)]
        layer_norm = torch.nn.LayerNorm([2, 5, 5])
        for tracing in [True, False]:
            m = self.checkGraphModeOp(
                layer_norm, data, "quantized::layer_norm", tracing
            )
            FileCheck().check_not("aten::layer_norm").run(m.graph)

    @override_qengines
    def test_group_norm(self):
        # [FUNCTION DOC] Checks group norm quantization (2 groups, 4 channels), traced and
        # scripted, under each quantization engine: the quantized group norm op must appear and no
        # float group norm remain.
        data = [[torch.rand((1, 4, 5, 5), dtype=torch.float)] for _ in range(2)]
        group_norm = torch.nn.GroupNorm(2, 4)
        for tracing in [True, False]:
            m = self.checkGraphModeOp(
                group_norm, data, "quantized::group_norm", tracing
            )
            FileCheck().check_not("aten::group_norm").run(m.graph)

    @override_qengines
    def test_instance_norm(self):
        # [FUNCTION DOC] Checks instance norm quantization for 1d, 2d and 3d (4 channels), traced
        # and scripted, under each quantization engine: the quantized instance norm op must appear
        # and no float instance norm remain.
        data_1d = [[torch.rand((1, 4, 5), dtype=torch.float)] for _ in range(2)]
        data_2d = [[torch.rand((1, 4, 5, 1), dtype=torch.float)] for _ in range(2)]
        data_3d = [[torch.rand((1, 4, 5, 1, 1), dtype=torch.float)] for _ in range(2)]
        data = {1: data_1d, 2: data_2d, 3: data_3d}
        instance_norm_modules = {
            1: torch.nn.InstanceNorm1d,
            2: torch.nn.InstanceNorm2d,
            3: torch.nn.InstanceNorm3d,
        }

        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            instance_norm = instance_norm_modules[dim](4)
            m = self.checkGraphModeOp(
                instance_norm, data[dim], "quantized::instance_norm", tracing
            )
            FileCheck().check_not("aten::instance_norm").run(m.graph)

    @skipIfNoFBGEMM
    def test_dequantize_tuple(self):
        # [FUNCTION DOC] Checks that conv outputs returned as a tuple of tensors can still be
        # quantized: runs the graph mode op check expecting a quantized conv2d, traced and
        # scripted. Skipped without fbgemm.
        """Make sure dequantize can support Tuple of tensor"""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two float convs applied to the same input; forward returns both
            # results as a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv1 and conv2 (3 to 3, kernel 3).
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Returns the tuple of conv1(x) and conv2(x).
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                return x1, x2

        for tracing in [True, False]:
            self.checkGraphModeOp(M(), self.img_data_2d, "quantized::conv2d", tracing)

    @skipIfNoFBGEMM
    def test_clamp(self):
        # [FUNCTION DOC] Checks clamp-like op quantization. For each of the clamp, hardtanh and
        # in-place hardtanh ops, traced and scripted, runs the graph mode op check on a conv
        # followed by a long chain of relu6, clamp and hardtanh variants and requires exactly one
        # input quantize node and one dequantize node, so the chain stays in the quantized domain.
        # Skipped without fbgemm.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Conv followed by a chain of clamp-like activations: ReLU6 module
            # (plain and in place), functional relu6, clamp as function and method, Hardtanh module
            # (plain and in place), functional hardtanh and in-place functional hardtanh.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float conv (2 to 2, kernel 2), ReLU6 modules
                # (plain and in place) and Hardtanh modules (plain and in place).
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu6 = torch.nn.ReLU6()
                self.relu6_ = torch.nn.ReLU6(True)
                self.hardtanh = torch.nn.Hardtanh()
                self.hardtanh_ = torch.nn.Hardtanh(inplace=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv and then the clamp-like activations in
                # sequence; the in-place module and functional calls mutate x without rebinding.
                # Returns x.
                x = self.conv(x)
                x = self.relu6(x)
                self.relu6_(x)
                x = F.relu6(x)
                x = torch.clamp(x, -3, 3)
                x = x.clamp(-2.5, 2.5)
                # x = x.clamp_(-2, 2)  # Enable when quantized `clamp_` is ready
                x = self.hardtanh(x)
                self.hardtanh_(x)
                x = F.hardtanh(x)
                F.hardtanh_(x)
                return x

        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float)]]
        options = itertools.product(
            ["aten::clamp", "aten::hardtanh", "aten::hardtanh_"], [True, False]
        )
        for op, tracing in options:
            m = self.checkGraphModeOp(M(), data, op, tracing)
            FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).run(
                m.graph
            )

            FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)

    def test_general_shape_ops(self):
        # [FUNCTION DOC] Checks that dequantize is propagated through every supported general shape
        # op (pooling, flatten, reshape, view, chunk, sort, dropout, relu variants, squeeze,
        # unsqueeze, repeat, stack, list and tuple construct and unpack, scalar add and mul
        # variants) without executing the model, which is only scripted. Inserts observers with a
        # scripted qconfig through the raw C++ pass, then convert_jit. Requires exactly 1 input
        # quantize node, 2 quantized conv2d ops, 1 dequantize node, and that quantized add-scalar
        # precedes quantized mul-scalar.
        """A test that checks dequantize will be swapped for
        all supported general shape ops like aten::flatten
        without actually checking for execution of these ops
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Large module chaining conv1, scalar add and mul variants, relus,
            # pooling, shape manipulation, list/tuple packing, dropout, sort, squeeze, unsqueeze,
            # detach, repeat, stack, and finally conv2. It is not meant to be executed, only
            # scripted and inspected.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates max pools (1d, 2d, 3d, kernel 3), Dropout, two
                # Conv2d (3 to 3, kernel 3) and a ReLU.
                super().__init__()
                self.maxpool1d = torch.nn.MaxPool1d(kernel_size=3)
                self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3)
                self.maxpool3d = torch.nn.MaxPool3d(kernel_size=3)
                self.dropout = torch.nn.Dropout()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the long op chain described by the inline
                # comments; the output of the final conv2 is returned. Shapes do not line up for
                # real execution.
                x = self.conv1(x)
                # add_scalar
                x = x + 3
                # mul_scalar
                x = x * 3
                # add_scalar_out
                x += 3
                # mul_scalar_out
                x *= 3
                # add_scalar_relu
                x = x + 3
                x = F.relu(x)
                # add_scalar_relu_out
                x += 3
                x = F.relu(x)
                # mul_scalar_relu
                x = x * 3
                x = F.relu(x)
                # mul_scalar_relu_out
                x *= 3
                x = F.relu(x)
                x = self.maxpool1d(x)
                x = self.maxpool2d(x)
                x = self.maxpool3d(x)
                x = torch.flatten(x)
                x = torch.max(x)
                x = torch.min(x)
                x = x.reshape([-1])
                x = x.resize_(1, 1, x.numel())
                x = x.view(-1)
                # prim::ListConstruct
                xs = [x, x]
                # prim::ListUnpack
                x, y = xs
                # prim::TupleConstruct
                xs = (x, x)
                # prim::TupleUnpack
                x, y = xs
                x = x.transpose(1, 2)
                x = x.contiguous()
                x, y = torch.chunk(x, 2)
                x = F.dropout(x)
                x = self.dropout(x)
                x, _ = torch.sort(x)
                x = x.permute(0, 2, 3, 1)
                x = torch.repeat_interleave(x, 3, 1)
                x = self.relu(x)
                x = F.relu(x)
                x.relu_()
                x = x.squeeze(0)
                x.squeeze_(0)
                x = torch.squeeze(x, 0)
                x = x.unsqueeze(0)
                x.unsqueeze_(0)
                x = torch.unsqueeze(x, 0)
                x = x.detach()
                x.detach_()
                x = x.repeat(4, 2)
                y = []
                y.append(x)
                z = torch.stack(y, 0)
                z = [z, z]
                x, _ = z
                x = self.conv2(x)
                return x

        data = torch.rand(1, 3, 10, 10)
        # This model is not executable since we just put all ops
        # in the same forward, therefore we only test scripting
        m = torch.jit.script(M())
        qconfig = script_qconfig(default_qconfig)
        # dummy data to suppress warning
        get_forward(qconfig.activation)(data)
        get_forward(qconfig.weight)(data)

        m = wrap_cpp_module(
            torch._C._jit_pass_insert_observers(
                m._c, "forward", {"": qconfig}, inplace=False
            )
        )
        m = convert_jit(m)
        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers and also successfully fused two quantized::conv2d
        # patterns
        # one quantize_per_tensor for input
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).run(
            m.graph
        )

        FileCheck().check_count("quantized::conv2d(", 2, exactly=True).run(m.graph)

        FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)

        FileCheck().check("quantized::add_scalar").check("quantized::mul_scalar").run(
            m.graph
        )

    def test_general_value_ops(self):
        # [FUNCTION DOC] Checks the patterns produced for general value ops (average and adaptive
        # pools, mean, interpolate variants, leaky relu, hardsigmoid, sigmoid, tanh) between two
        # convs, with the model only scripted. After the observer insertion pass, convert_jit with
        # debug on must show the expected total of per-tensor quantize nodes (one for output plus
        # per-op counts from a table, minus 4 for folded prepacks); convert_jit without debug must
        # show 1 quantize node and 2 quantized conv2d ops followed by a dequantize.
        """ A test that checks correct patterns are produced for
        all supported general value ops like aten::avg_pool2d \
        without actually checking for execution of these ops
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module of a conv, then a long sequence of general value ops
            # (pooling modules and functions, mean, interpolate and upsample variants, activations
            # with in-place and function forms), and a second use of the same conv. Not executable,
            # only scripted.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv (3 to 3, kernel 3), average pool
                # 1d/2d/3d, adaptive average pool 1d/2d/3d, LeakyReLU, Hardsigmoid, Sigmoid and
                # Tanh submodules.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.avg_pool1d = torch.nn.AvgPool1d(3)
                self.avg_pool2d = torch.nn.AvgPool2d(3)
                self.avg_pool3d = torch.nn.AvgPool3d(3)
                self.adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d(1)
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                self.leaky_relu = torch.nn.LeakyReLU()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv and each listed op in order and returns
                # the output of the second conv call.
                x = self.conv(x)
                x = self.avg_pool1d(x)
                x = self.avg_pool2d(x)
                x = self.avg_pool3d(x)
                x = self.adaptive_avg_pool1d(x)
                x = self.adaptive_avg_pool2d(x)
                x = self.adaptive_avg_pool3d(x)
                x = F.avg_pool1d(x, 3)
                x = F.avg_pool2d(x, 3)
                x = F.avg_pool3d(x, 3)
                x = F.adaptive_avg_pool1d(x, (1))
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = F.adaptive_avg_pool3d(x, (1, 1, 1))
                x = torch.mean(x)
                x = torch.mean(x, [2, 3], False)
                x = x.mean()
                x = x.mean([2, 3], True)
                # interpolate node will introduce 3 quantize_per_tensor ops
                x = F.interpolate(x, 4, mode="nearest")  # interpolate node
                x = F.upsample(x, (32, 32))  # interpolate node
                x = F.upsample_nearest(x, (32, 32))  # interpolate node
                x = F.interpolate(x, 4, mode="linear")  # common node
                x = F.upsample_bilinear(x, (32, 32))  # common node
                x = self.leaky_relu(x)
                x = F.leaky_relu(x)
                x.leaky_relu_()
                x = self.hardsigmoid(x)
                x = F.hardsigmoid(x)
                x.hardsigmoid_()
                x = self.sigmoid(x)
                x = torch.sigmoid(x)
                # F.sigmoid is deprecated
                x = x.sigmoid()
                x.sigmoid_()
                x = self.tanh(x)
                # F.tanh is deprecated
                x = torch.tanh(x)
                x = x.tanh()
                x.tanh_()
                x = self.conv(x)
                return x

        # This model is not executable since we just put all ops
        # in the same forward, therefore we only test scripting
        m = torch.jit.script(M())
        qconfig = script_qconfig(default_qconfig)
        # dummy data to suppress warning
        data = torch.rand(1, 3, 10, 10)
        get_forward(qconfig.activation)(data)
        get_forward(qconfig.weight)(data)

        m = wrap_cpp_module(
            torch._C._jit_pass_insert_observers(
                m._c, "forward", {"": qconfig}, inplace=False
            )
        )
        # Checking the model before finalize contain unfused patterns
        # that numerically matches the model after quantize by checking
        # number of aten::quantize_per_tensor functions
        # conv has 3 quantize_per_tensor for activations and 1 for weight
        # and for N general value op between conv we should have

        # N + 1 quantize_per_tensor between these ops
        m1 = convert_jit(m, debug=True)
        # NB: This Needs to be updated when we add more ops to test
        # mapping from number of quant for the op to the number of these ops
        # for example, for `3` in the key means for this type of op
        # we'll have 3 quantize_per_tensor
        num_op_by_num_quant = {1: 32, 2: 2, 3: 3}
        num_quantize_per_tensor = 1  # for output
        for num_quant, num_op in num_op_by_num_quant.items():
            num_quantize_per_tensor += num_op * num_quant
        num_quantize_per_tensor -= 4  # constant propagation removes some prepacks
        FileCheck().check_count(
            "aten::quantize_per_tensor(", num_quantize_per_tensor, exactly=True
        ).run(m1.graph)

        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers and also successfully fused two quantized::conv2d
        # patterns
        # one quantize_per_tensor for input
        m2 = convert_jit(m, debug=False)
        FileCheck().check_count("aten::quantize_per_tensor(", 1, exactly=True).run(
            m2.graph
        )
        FileCheck().check_count("quantized::conv2d(", 2, exactly=True).check(
            "aten::dequantize("
        ).run(m2.graph)

    @override_qengines
    def test_conv_with_benchmark_flag(self):
        # [FUNCTION DOC] Verifies convs still get quantized when cudnn benchmark flags are
        # active. Does nothing unless the active engine is qnnpack; otherwise traces a one-conv
        # Sequential with cudnn enabled, prepares and converts it with the qnnpack default
        # qconfig, and requires a quantized conv2d in the graph. Runs for each engine through the
        # override decorator.
        r"""Verifies that convolutions get quantized when
        torch.backends.cudnn.benchmark is enabled
        """
        if not qengine_is_qnnpack():
            return
        with torch.backends.cudnn.flags(enabled=True):
            m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
            m.eval()
            m = torch.jit.trace(m, torch.rand(4, 1, 4, 4))
            qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
            prepared_model = torch.ao.quantization.prepare_jit(m, {"": qconfig})
            prepared_model(torch.rand(4, 1, 4, 4))
            converted_model = torch.ao.quantization.convert_jit(prepared_model)
            FileCheck().check("quantized::conv2d").run(converted_model.graph)

    @skipIfNoFBGEMM
    def test_cat_linear(self):
        # [FUNCTION DOC] Checks that two linears fed by a concatenation are both quantized: scripts
        # the model, prepares, calibrates once with random inputs, converts, and requires two
        # quantized linear ops in the graph. Skipped without fbgemm.
        class LinearModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Concatenates two inputs and applies functional linear twice with
            # the same random plain-tensor weight (5 by 5); returns both linear outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the random weight tensor attribute.
                super().__init__()
                self.weight = torch.randn(5, 5)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Concatenates x and y, applies linear to get b, applies
                # linear to b to get c, returns b and c.
                a = torch.cat([x, y])
                b = F.linear(a, self.weight)
                c = F.linear(b, self.weight)
                return b, c

        model = LinearModel().eval()
        qconfig = {"": default_qconfig}
        float_model = torch.jit.script(model)
        prepared_model = prepare_jit(float_model, qconfig)
        prepared_model(torch.rand(5, 5), torch.rand(5, 5))
        converted_model = convert_jit(prepared_model)
        FileCheck().check("quantized::linear").check("quantized::linear").run(
            converted_model.graph
        )


class TestQuantizeDynamicJitPasses(QuantizationTestCase):
    # [CLASS DOC] Tests for the passes behind dynamic quantization in TorchScript
    # (prepare_dynamic_jit, quantize_dynamic_jit, convert_dynamic_jit): observer placement,
    # quant/dequant insertion for linear, shared weights, conditionals, weight observer parameter
    # values and fp16 variants.
    def test_prepare_dynamic(self):
        # [FUNCTION DOC] Checks observers inserted by prepare_dynamic_jit on a linear for the fp16
        # and default dynamic qconfigs. The linear always has one weight observer; for fp16 a
        # placeholder observer must appear in the linear graph; for the default dynamic config the
        # root must have one input observer that precedes the call of fc.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single Linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc applied to x.
                return self.fc(x)

        model = torch.jit.script(M())
        for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
            m = prepare_dynamic_jit(model, {"": qconfig})

            # observer for weight
            if len(attrs_with_prefix(m.fc, "_observer_")) != 1:
                raise AssertionError(
                    f"Expected 1 observer, got {len(attrs_with_prefix(m.fc, '_observer_'))}"
                )

            if qconfig == float16_dynamic_qconfig:
                observer_name = 'PlaceholderObserver = prim::GetAttr[name="_observer_'
                FileCheck().check(observer_name).run(m.fc.graph)
            else:
                # for input of FC for dynamic quant
                if len(attrs_with_prefix(m, "_observer_")) != 1:
                    raise AssertionError(
                        f"Expected 1 observer, got {len(attrs_with_prefix(m, '_observer_'))}"
                    )
                observer_name = 'Observer = prim::GetAttr[name="_observer_'
                FileCheck().check(observer_name).check(
                    'prim::GetAttr[name="fc"]'
                ).check("prim::CallMethod").check_not(observer_name).run(m.graph)

    def test_prepare_dynamic_child_qconfig(self):
        # [FUNCTION DOC] Checks dynamic preparation with a qconfig only for the nested sub.fc.
        # Requires 1 observer at the root (input of sub), 0 on conv, 0 on sub (observed at the
        # outermost call site), 1 on fc (weight), and uses FileCheck to confirm the observer
        # precedes the call into sub and none follows.
        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Wraps a single Linear named fc.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc applied to x.
                return self.fc(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: A conv followed by a Sub submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv (3 to 5, kernel 3) and the Sub submodule.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then sub and returns the result.
                return self.sub(self.conv(x))

        m = torch.jit.script(M())
        # only quantize child module.
        m = prepare_dynamic_jit(m, {"sub.fc": default_dynamic_qconfig})

        # input of sub for dynamic quant
        if len(attrs_with_prefix(m, "_observer_")) != 1:
            raise AssertionError(
                f"Expected 1 observer, got {len(attrs_with_prefix(m, '_observer_'))}"
            )
        # not quantized
        if len(attrs_with_prefix(m.conv, "_observer_")) != 0:
            raise AssertionError(
                f"Expected 0 observers, got {len(attrs_with_prefix(m.conv, '_observer_'))}"
            )
        # no observers since we observe in the outer most call site
        if len(attrs_with_prefix(m.sub, "_observer_")) != 0:
            raise AssertionError(
                f"Expected 0 observers, got {len(attrs_with_prefix(m.sub, '_observer_'))}"
            )
        # weight of linear
        if len(attrs_with_prefix(m.sub.fc, "_observer_")) != 1:
            raise AssertionError(
                f"Expected 1 observer, got {len(attrs_with_prefix(m.sub.fc, '_observer_'))}"
            )
        FileCheck().check('prim::GetAttr[name="sub').check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("prim::CallMethod").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(m.graph)

    def test_insert_quant_dequant_linear_dynamic(self):
        # [FUNCTION DOC] Checks dynamic quant/dequant insertion for two linears with per-channel
        # and default dynamic qconfigs and debug on. Requires two submodules and uses FileCheck to
        # confirm the order: choose-qparams, per-tensor quantize and dequantize for each
        # activation, then the weight quantize (per-channel or per-tensor) followed by dequantize.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two float Linear layers applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1 and fc2 (5 to 5, float).
                super().__init__()
                self.fc1 = torch.nn.Linear(5, 5).float()
                self.fc2 = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1 then fc2 and returns the result.
                x = self.fc1(x)
                return self.fc2(x)

        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            qconfig = (
                per_channel_dynamic_qconfig
                if is_per_channel is True
                else default_dynamic_qconfig
            )
            m = quantize_dynamic_jit(m, {"": qconfig}, debug=True)
            if len(m._modules._c.items()) != 2:
                raise AssertionError("Expected to have two submodule of linear")

            wt_quant_func = (
                "aten::quantize_per_channel"
                if is_per_channel
                else "aten::quantize_per_tensor"
            )
            act_quant_func = "aten::quantize_per_tensor"
            # quantizing activations
            FileCheck().check("aten::_choose_qparams_per_tensor").check_next(
                act_quant_func
            ).check_next("aten::dequantize").check(
                "aten::_choose_qparams_per_tensor"
            ).check_next(act_quant_func).check_next("aten::dequantize").check(
                wt_quant_func
            ).check_next("aten::dequantize").check_not(wt_quant_func).check(
                "return"
            ).run(m.graph)

    @override_qengines
    def test_dynamic_multi_op(self):
        # [FUNCTION DOC] Checks that only linear is dynamically quantized when other ops are
        # present: for traced and scripted models it requires the dynamic quantized linear op and
        # an add op still in the graph. Runs under each quantization engine.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Adds 5 to the input, then applies a float Linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Linear (5 to 5) as fc1 in float dtype.
                super().__init__()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 5 to x and returns fc1 of the result.
                x = x + 5
                return self.fc1(x)

        x = torch.randn(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(
                M(), x, "quantized::linear_dynamic", tracing=tracing, dynamic=True
            )
            # add op is not dynamically quantized.
            FileCheck().check("aten::add").run(model.graph)

    @override_qengines
    def test_dynamic_quant_multi_uses(self):
        # [FUNCTION DOC] Checks that an input with multiple users (size queries plus the linear)
        # gets no choose-qparams node while the linear is still dynamically quantized, traced and
        # scripted, under each engine.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear on the input plus two size queries of the input, all
            # returned as a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates float Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads x.size twice and returns fc(x) with the two sizes.
                size1 = x.size()
                size2 = x.size()
                return self.fc(x), size1, size2

        x = torch.randn(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(
                M(), x, "quantized::linear_dynamic", tracing=tracing, dynamic=True
            )
            FileCheck().check_not("aten::_choose_qparams_per_tensor").run(model.graph)

    @override_qengines
    def test_dynamic_shared_weights(self):
        # [FUNCTION DOC] Checks dynamic quantization when a weight parameter is shared
        # between a Linear submodule and a functional linear. With the qconfig on mod1
        # only, one dynamic linear is expected; with the qconfig on the root, two. Also
        # checks the quantize_dynamic_jit result equals the manual prepare, run and
        # convert_dynamic_jit result.
        class myMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Wraps a Linear whose weight is replaced by the given shared
            # weight parameter.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Creates Linear (5 to 5) and assigns the given weight as
                # its weight.
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.linear.weight = weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear applied to x.
                return self.linear(x)

        class DynamicModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Owns a shared weight parameter (ones, 5 by 5), a myMod using it
            # as mod1, and a functional linear using the same weight.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ones weight parameter and mod1 built from it.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(5, 5))
                self.mod1 = myMod(self.weight)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies mod1 to x, then functional linear with the shared
                # weight, and returns that.
                y = self.mod1(x)
                z = torch.nn.functional.linear(y, self.weight)
                return z

        model = torch.jit.script(DynamicModel()).eval()
        data = torch.randn(5, 5, dtype=torch.float)
        quant_ops = ["mod1", ""]
        counts = [1, 2]
        for op, count in zip(quant_ops, counts):
            qconfig_dict = {op: default_dynamic_qconfig}
            m1 = quantize_dynamic_jit(model, qconfig_dict)
            out_graph = m1(data)

            FileCheck().check_count(
                "quantized::linear_dynamic(", count, exactly=True
            ).check_not("aten::_choose_qparams_per_tensor").run(m1.graph)

            # Explicitly call forward on model before convert
            m2 = prepare_dynamic_jit(model, qconfig_dict)
            m2(data)
            m2 = convert_dynamic_jit(m2, debug=False)
            out_ref = m2(data)
            self.assertEqual(out_graph, out_ref)

    @override_qengines
    def test_dynamic_with_if(self):
        # [FUNCTION DOC] Checks dynamic quantization of linears inside conditional branches. Traced
        # and scripted models must contain exactly 2 dynamic quantized linear ops and no
        # choose-qparams node. Then computes reference scale and zero point of each weight with a
        # scripted weight observer and compares them with the scale and zero-point attributes
        # stored on the converted res1 and res2 modules (value names like weight.2 and weight.4
        # come from the graph numbering).
        class Res(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a ones weight parameter (5 by 5); forward applies
            # functional linear in both branches of a boolean condition.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ones weight parameter.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x: torch.Tensor, cond: bool) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns functional linear of x with weight regardless
                # of cond.
                if cond:
                    return torch.nn.functional.linear(x, self.weight)
                else:
                    return torch.nn.functional.linear(x, self.weight)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two Res blocks; res1 called with true and res2 with false.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates res1 and res2.
                super().__init__()
                self.res1 = Res()
                self.res2 = Res()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies res1 with true, then res2 with false, and returns
                # the result.
                x = self.res1(x, True)
                x = self.res2(x, False)
                return x

        model = torch.jit.script(M()).eval()
        data = torch.randn(5, 5, dtype=torch.float)
        qconfig_dict = {"": default_dynamic_qconfig}
        for tracing in [True, False]:
            m1 = self.checkGraphModeOp(
                M(), data, "quantized::linear_dynamic", tracing=tracing, dynamic=True
            )
            FileCheck().check_count(
                "quantized::linear_dynamic(", 2, exactly=True
            ).check_not("aten::_choose_qparams_per_tensor").run(m1.graph)

        # Check to make sure weight observers run correctly
        ref_qparams = []
        qconfig = script_qconfig(default_dynamic_qconfig)
        wt_module = wrap_cpp_module(qconfig.weight)
        for wt in [model.res1.weight, model.res2.weight]:
            wt_module(wt)
            qparams = wt_module.calculate_qparams()
            ref_qparams.append((qparams[0].item(), qparams[1].item()))

        m2 = quantize_dynamic_jit(model, qconfig_dict, debug=True)
        graph_params = []
        for x, obs in m2._modules._c.items():
            if x == "res1":
                graph_params.append(
                    (
                        obs.getattr("weight.2_scale_0"),
                        obs.getattr("weight.2_zero_point_0"),
                    )
                )
            elif x == "res2":
                graph_params.append(
                    (
                        obs.getattr("weight.4_scale_0"),
                        obs.getattr("weight.4_zero_point_0"),
                    )
                )
        self.assertEqual(ref_qparams, graph_params)

    def test_dynamic_weight_observer(self):
        # [FUNCTION DOC] Checks that weight observers in dynamic quantization compute the same
        # scale and zero point as running the default dynamic weight observer on each weight in
        # eager. For traced and scripted models, compares reference values with the
        # scale/zero-point attributes saved on each converted submodule (the value name numeral
        # depends on graph numbering: 2 for the first linear when traced, otherwise 1).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two float Linear layers applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc and fc2 (5 to 5, float).
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()
                self.fc2 = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc then fc2 and returns the result.
                x = self.fc(x)
                return self.fc2(x)

        qconfig_dict = {"": default_dynamic_qconfig}
        eager_model = M().eval()
        for tracing in [True, False]:
            x = torch.rand(5, 5)
            model = get_script_module(eager_model, tracing, x)
            ref_qparams = []
            for wt in [model.fc.weight, model.fc2.weight]:
                wt_module = default_dynamic_qconfig.weight()
                wt_module(wt)
                qparams = wt_module.calculate_qparams()
                ref_qparams.append((qparams[0].item(), qparams[1].item()))
            model = quantize_dynamic_jit(model, qconfig_dict, debug=True)
            graph_qparams = []
            for x, obs in model._modules._c.items():
                n = 2 if x == "fc" and tracing else 1
                graph_qparams.append(
                    (
                        obs.getattr(f"weight.{n}_scale_0"),
                        obs.getattr(f"weight.{n}_zero_point_0"),
                    )
                )
            self.assertEqual(ref_qparams, graph_qparams)

    def test_convert_dynamic_fp16(self):
        # [FUNCTION DOC] Checks converting with the fp16 dynamic qconfig and debug on: the weight
        # is saturated to half precision, a float linear is kept, and no dequantize or quantize
        # nodes appear.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single Linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc applied to x.
                return self.fc(x)

        m = torch.jit.script(M())
        m = quantize_dynamic_jit(m, {"": float16_dynamic_qconfig}, debug=True)
        FileCheck().check("aten::_saturate_weight_to_fp16").check(
            "aten::linear"
        ).check_not("aten::dequantize").check_not("aten::quantize").run(m.graph)

    def test_quantize_dynamic_fp16(self):
        # [FUNCTION DOC] Checks quantize_dynamic_jit without debug and with the fp16 dynamic
        # qconfig: the graph must contain the dynamic fp16 quantized linear and no float linear,
        # dequantize or quantize nodes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single Linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates Linear (5 to 5) as fc.
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc applied to x.
                return self.fc(x)

        m = torch.jit.script(M())
        m = quantize_dynamic_jit(m, {"": float16_dynamic_qconfig})

        FileCheck().check("quantized::linear_dynamic_fp16").check_not(
            "aten::linear"
        ).check_not("aten::dequantize").check_not("aten::quantize").run(m.graph)


class TestQuantizeDynamicJitOps(QuantizationTestCase):
    # [CLASS DOC] Tests that graph-mode post training dynamic quantization produces the expected
    # quantized operator end to end for individual ops (linear, embedding bag, LSTM and similar),
    # via the checkGraphModeOp helper with dynamic set.
    """Test graph mode post training dynamic quantization works
    for individual ops end to end.
    """

    @override_qengines
    def test_linear(self):
        # [FUNCTION DOC] Checks dynamic linear quantization for the Linear module and for a
        # functional linear with and without bias, traced and scripted, requiring the dynamic
        # quantized linear op. Runs under each quantization engine.
        class FunctionalLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Functional linear with externally given weight and bias (bias may
            # be None) stored as plain attributes.
            def __init__(self, weight, bias):
                # [FUNCTION DOC] HELPER: Stores the weight and bias tensors.
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional linear of x with the stored weight
                # and bias.
                return F.linear(x, self.weight, self.bias)

        x = torch.rand(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(
                torch.nn.Linear(5, 5),
                x,
                "quantized::linear_dynamic",
                tracing=tracing,
                dynamic=True,
            )

        weight = torch.rand(5, 5)
        b = torch.rand(5)
        for tracing, has_bias in itertools.product([True, False], [True, False]):
            bias = b if has_bias else None
            model = self.checkGraphModeOp(
                FunctionalLinear(weight, bias),
                x,
                "quantized::linear_dynamic",
                tracing=tracing,
                dynamic=True,
            )

    @skipIfNoFBGEMM
    def test_embedding_bag(self):
        # [FUNCTION DOC] Checks static quantization of two sparse embedding bags sharing one weight
        # tensor: for traced and scripted versions, uses a 4-bit placeholder qconfig for embedding1
        # and a byte qconfig for embedding2, runs prepare_jit and convert_jit, requires the 4-bit
        # rowwise-offsets op followed by the byte rowwise-offsets op in the graph, and then runs
        # the quantized model on the dummy inputs. Skipped without fbgemm.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two sum-mode sparse EmbeddingBags (10 embeddings, dim 12) both
            # initialised from one weight tensor; forward returns both lookups as a tuple.
            def __init__(self, weights):
                # [FUNCTION DOC] HELPER: Creates embedding1 and embedding2 from the given weights,
                # with include-last-offset and sparse enabled.
                super().__init__()
                self.embedding1 = torch.nn.EmbeddingBag(
                    num_embeddings=10,
                    embedding_dim=12,
                    include_last_offset=True,
                    sparse=True,
                    _weight=weights,
                    mode="sum",
                )

                self.embedding2 = torch.nn.EmbeddingBag(
                    num_embeddings=10,
                    embedding_dim=12,
                    include_last_offset=True,
                    sparse=True,
                    _weight=weights,
                    mode="sum",
                )

            def forward(self, indices1, offsets1, indices2, offsets2):
                # [FUNCTION DOC] HELPER: Looks up indices1 with offsets1 in embedding1 and indices2
                # with offsets2 in embedding2, returns both results.
                e1 = self.embedding1(indices1, offsets1)
                e2 = self.embedding2(indices2, offsets2)
                return e1, e2

        weights = torch.randn(10, 12, dtype=torch.float32)
        module = M(weights)

        indices = torch.tensor(
            [
                9,
                6,
                5,
                7,
                8,
                8,
                9,
                2,
                8,
                6,
                6,
                9,
                1,
                6,
                8,
                8,
                3,
                2,
                3,
                6,
                3,
                6,
                5,
                7,
                0,
                8,
                4,
                6,
                5,
                8,
                2,
                3,
            ]
        )
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        dummy_inputs = (indices, offsets, indices, offsets)
        for trace in [True, False]:
            if trace:
                m = torch.jit.trace(module, dummy_inputs)
            else:
                m = torch.jit.script(module)
            int4_qconfig = QConfig(
                activation=PlaceholderObserver.with_args(
                    dtype=torch.float, custom_op_name="embedding_bag_4bit"
                ),
                weight=PlaceholderObserver.with_args(
                    custom_op_name="embedding_bag_4bit"
                ),
            )
            int8_qconfig = QConfig(
                activation=PlaceholderObserver.with_args(
                    dtype=torch.float, custom_op_name="embedding_bag_byte"
                ),
                weight=PlaceholderObserver.with_args(
                    custom_op_name="embedding_bag_byte"
                ),
            )
            m = prepare_jit(m, {"embedding1": int4_qconfig, "embedding2": int8_qconfig})
            m = convert_jit(m)
            FileCheck().check("quantized::embedding_bag_4bit_rowwise_offsets").check(
                "quantized::embedding_bag_byte_rowwise_offsets"
            ).run(m.graph)
            m(*dummy_inputs)

    # Ensure that attempting to quantize an EmbeddingBag throws an error if
    # padding_idx is not None
    @skipIfNoFBGEMM
    def test_embedding_bag_padding_idx_error(self):
        # [FUNCTION DOC] Checks that quantizing an EmbeddingBag with a non-None padding index
        # fails. For traced and scripted modules and for 4-bit and byte qconfigs, prepare_jit
        # succeeds but convert_jit must raise a RuntimeError whose message says the padding_idx
        # input is expected to be None. Skipped without fbgemm.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Single sum-mode sparse EmbeddingBag (10 embeddings, dim 12,
            # include last offset, padding index 0) built from a supplied weight.
            def __init__(self, weights):
                # [FUNCTION DOC] HELPER: Creates the embedding attribute from the given weights
                # with padding_idx 0.
                super().__init__()
                self.embedding = torch.nn.EmbeddingBag(
                    num_embeddings=10,
                    embedding_dim=12,
                    include_last_offset=True,
                    sparse=True,
                    _weight=weights,
                    mode="sum",
                    padding_idx=0,
                )

            def forward(self, indices, offsets):
                # [FUNCTION DOC] HELPER: Returns the embedding lookup for the indices and offsets.
                e = self.embedding(indices, offsets)
                return e

        weights = torch.randn(10, 12, dtype=torch.float32)
        module = M(weights)

        indices = torch.tensor([0, 1, 2, 3, 4])
        offsets = torch.tensor([0, 2, 5])
        dummy_inputs = (indices, offsets)

        int4_qconfig = QConfig(
            activation=PlaceholderObserver.with_args(
                dtype=torch.float, custom_op_name="embedding_bag_4bit"
            ),
            weight=PlaceholderObserver.with_args(custom_op_name="embedding_bag_4bit"),
        )
        int8_qconfig = QConfig(
            activation=PlaceholderObserver.with_args(
                dtype=torch.float, custom_op_name="embedding_bag_byte"
            ),
            weight=PlaceholderObserver.with_args(custom_op_name="embedding_bag_byte"),
        )

        error_msg = r"Expected aten::embedding_bag padding_idx input to be None"
        for trace, qconfig in itertools.product(
            [True, False], [int4_qconfig, int8_qconfig]
        ):
            if trace:
                m = torch.jit.trace(module, dummy_inputs)
            else:
                m = torch.jit.script(module)
            m = prepare_jit(m, {"embedding": qconfig})
            with self.assertRaisesRegex(RuntimeError, error_msg):
                m = convert_jit(m)


class TestQuantizeJit(QuantizationTestCase):
    # [CLASS DOC] Compares TorchScript graph-mode quantization (quantize_jit and
    # quantize_dynamic_jit on traced and scripted models) with eager-mode quantization for small
    # standard models from common_quantization: linear, conv, transposed conv, conv-bn, nested and
    # skip-quant models, and dynamic linear. Each test copies the eager model weights into the
    # model to script so outputs can be compared with assertEqual.
    @override_qengines
    def test_single_linear(self):
        # [FUNCTION DOC] Compares eager and graph mode static quantization of a single linear layer
        # under each quantization engine. Copies weights from the annotated eager model into the
        # plain model, quantizes the eager model with the calibration data, then quantizes the
        # traced and scripted versions with quantize_jit (not in place) and requires equal outputs.
        r"""Compare the result of quantizing single linear layer in
        eager mode and graph mode
        """
        # eager mode
        annotated_linear_model = AnnotatedSingleLayerLinearModel(
            torch.backends.quantized.engine
        ).eval()
        linear_model = SingleLayerLinearModel().eval()
        # copy the weight from eager mode so that we can
        # compare the result of the two quantized models later
        linear_model.fc1.weight = torch.nn.Parameter(
            annotated_linear_model.fc1.module.weight.detach()
        )
        linear_model.fc1.bias = torch.nn.Parameter(
            annotated_linear_model.fc1.module.bias.detach()
        )
        model_eager = quantize(
            annotated_linear_model, test_only_eval_fn, [self.calib_data]
        )

        qconfig_dict = {"": get_default_qconfig(torch.backends.quantized.engine)}
        model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
        model_script = torch.jit.script(linear_model)
        result_eager = model_eager(self.calib_data[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.calib_data],
                inplace=False,
            )
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @skipIfNoFBGEMM
    def test_observer_with_ignored_function(self):
        # [FUNCTION DOC] Checks static quantization with three qconfigs (plain observers, histogram
        # activation observer, per-channel weight observer) matches eager mode for traced and
        # scripted single-linear models. The test name mentions an ignored function but the body
        # does not use one. Skipped without fbgemm.
        r"""Test observers with ignored function and make sure it works in
        graph mode
        """
        # eager mode
        annotated_linear_model = AnnotatedSingleLayerLinearModel("fbgemm").eval()
        for qconfig in [
            QConfig(activation=default_observer, weight=default_weight_observer),
            QConfig(
                activation=default_histogram_observer, weight=default_weight_observer
            ),
            QConfig(
                activation=default_observer, weight=default_per_channel_weight_observer
            ),
        ]:
            annotated_linear_model.qconfig = qconfig
            linear_model = SingleLayerLinearModel().eval()
            # copy the weight from eager mode so that we can
            # compare the result of the two quantized models later
            linear_model.fc1.weight = torch.nn.Parameter(
                annotated_linear_model.fc1.module.weight.detach()
            )
            linear_model.fc1.bias = torch.nn.Parameter(
                annotated_linear_model.fc1.module.bias.detach()
            )
            model_eager = quantize(
                annotated_linear_model, test_only_eval_fn, [self.calib_data]
            )

            qconfig_dict = {"": qconfig}
            model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
            model_script = torch.jit.script(linear_model)
            result_eager = model_eager(self.calib_data[0][0])
            for model_under_test in [model_traced, model_script]:
                model_quantized = quantize_jit(
                    model_under_test,
                    qconfig_dict,
                    test_only_eval_fn,
                    [self.calib_data],
                    inplace=False,
                )
                self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines
    def test_conv(self):
        # [FUNCTION DOC] Compares eager and graph mode static quantization of a conv model under
        # each quantization engine: copies the conv weight, quantizes eager and traced/scripted
        # versions with the image data, and requires equal outputs.
        r"""Compare the result of quantizing conv layer in
        eager mode and graph mode
        """
        # eager mode
        annotated_conv_model = AnnotatedConvModel(
            torch.backends.quantized.engine
        ).eval()
        conv_model = ConvModel().eval()
        # copy the weight from eager mode so that we can
        # compare the result of the two quantized models later
        conv_model.conv.weight = torch.nn.Parameter(
            annotated_conv_model.conv.weight.detach()
        )
        model_eager = quantize(
            annotated_conv_model, test_only_eval_fn, [self.img_data_2d]
        )
        qconfig_dict = {"": get_default_qconfig(torch.backends.quantized.engine)}
        model_traced = torch.jit.trace(conv_model, self.img_data_2d[0][0])
        model_script = torch.jit.script(conv_model)
        result_eager = model_eager(self.img_data_2d[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.img_data_2d],
                inplace=False,
            )
            self.assertEqual(model_quantized(self.img_data_2d[0][0]), result_eager)

    @override_qengines
    def test_conv_transpose(self):
        # [FUNCTION DOC] Compares eager and graph mode static quantization of a transposed conv
        # model. Returns immediately unless the active engine is qnnpack; otherwise copies weights,
        # quantizes both ways and requires equal outputs for the traced and scripted models.
        r"""Compare the result of quantizing conv_transpose layer in
        eager mode and graph mode
        """
        if not qengine_is_qnnpack():
            return  # Currently only qnnpack is supported
        # eager mode
        annotated_conv_model = AnnotatedConvTransposeModel(
            torch.backends.quantized.engine
        ).eval()
        conv_model = ConvTransposeModel().eval()
        # copy the weight from eager mode so that we can
        # compare the result of the two quantized models later
        conv_model.conv.weight = torch.nn.Parameter(
            annotated_conv_model.conv.weight.detach()
        )
        model_eager = quantize(
            annotated_conv_model, test_only_eval_fn, [self.img_data_2d]
        )
        qconfig_dict = {"": get_default_qconfig(torch.backends.quantized.engine)}
        model_traced = torch.jit.trace(conv_model, self.img_data_2d[0][0])
        model_script = torch.jit.script(conv_model)
        result_eager = model_eager(self.img_data_2d[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.img_data_2d],
                inplace=False,
            )
            self.assertEqual(model_quantized(self.img_data_2d[0][0]), result_eager)

    @override_qengines
    def test_conv_bn(self):
        # [FUNCTION DOC] Compares eager quantization of a conv-bn model (after fusing conv and
        # bn in eager) with quantize_jit of the scripted conv-bn model (the bn is folded by
        # the pipeline), copying the conv weight first; requires equal outputs for the first
        # image batch.
        r"""Compare the result of quantizing conv + bn layer in
        eager mode and graph mode
        """
        # eager mode
        conv_model = AnnotatedConvBnModel().eval()
        conv_model_to_script = ConvBnModel().eval()
        # copy the weight from eager mode so that we can
        # compare the result of the two quantized models later
        conv_model_to_script.conv.weight = torch.nn.Parameter(
            conv_model.conv.weight.detach()
        )
        fuse_modules(conv_model, ["conv", "bn"], inplace=True)
        model_eager = quantize(conv_model, test_only_eval_fn, [self.img_data_2d])
        qconfig_dict = {"": default_qconfig}
        model_script = quantize_jit(
            torch.jit.script(conv_model_to_script),
            qconfig_dict,
            test_only_eval_fn,
            [self.img_data_2d],
            inplace=False,
        )
        result_eager = model_eager(self.img_data_2d[0][0])
        result_script = model_script(self.img_data_2d[0][0])
        self.assertEqual(result_eager, result_script)

    @override_qengines
    def test_nested(self):
        # [FUNCTION DOC] Compares eager and graph mode quantization of a nested model with
        # submodules. Copies all weights across, uses a qconfig for sub2.fc1 (per-channel if the
        # engine is fbgemm, else default) and fc3, quantizes the traced and scripted models and
        # requires outputs equal to the eager quantized output.
        # Eager mode
        eager_model = AnnotatedNestedModel(torch.backends.quantized.engine).eval()

        # Graph mode
        script_model = NestedModel().eval()
        # Copy weights for eager_model
        script_model.sub1.fc.weight = torch.nn.Parameter(
            eager_model.sub1.fc.weight.detach()
        )
        script_model.sub1.fc.bias = torch.nn.Parameter(
            eager_model.sub1.fc.bias.detach()
        )
        script_model.sub2.fc1.weight = torch.nn.Parameter(
            eager_model.sub2.fc1.module.weight.detach()
        )
        script_model.sub2.fc1.bias = torch.nn.Parameter(
            eager_model.sub2.fc1.module.bias.detach()
        )
        script_model.sub2.fc2.weight = torch.nn.Parameter(
            eager_model.sub2.fc2.weight.detach()
        )
        script_model.sub2.fc2.bias = torch.nn.Parameter(
            eager_model.sub2.fc2.bias.detach()
        )
        script_model.fc3.weight = torch.nn.Parameter(
            eager_model.fc3.module.weight.detach()
        )
        script_model.fc3.bias = torch.nn.Parameter(eager_model.fc3.module.bias.detach())

        model_eager = quantize(eager_model, test_only_eval_fn, [self.calib_data])
        qconfig_dict = {
            "sub2.fc1": default_per_channel_qconfig
            if qengine_is_fbgemm()
            else default_qconfig,
            "fc3": default_qconfig,
        }
        model_traced = torch.jit.trace(script_model, self.calib_data[0][0])
        model_script = torch.jit.script(script_model)
        result_eager = model_eager(self.calib_data[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.calib_data],
                inplace=False,
            )
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines
    def test_skip_quant(self):
        # [FUNCTION DOC] Checks quantization of a model with a submodule explicitly excluded by a
        # None qconfig for fc. Copies weights from the eager annotated model, fuses modules in
        # eager, quantizes eager and graph versions, and requires equal outputs for traced and
        # scripted models.
        """Test None qconfig"""
        # Eager mode
        eager_model = AnnotatedSkipQuantModel(torch.backends.quantized.engine).eval()

        # Graph mode
        script_model = SkipQuantModel().eval()
        # Copy weights for eager_model
        script_model.sub.fc1.weight = torch.nn.Parameter(
            eager_model.sub.module.fc1.weight.detach()
        )
        script_model.sub.fc1.bias = torch.nn.Parameter(
            eager_model.sub.module.fc1.bias.detach()
        )
        script_model.sub.fc2.weight = torch.nn.Parameter(
            eager_model.sub.module.fc2.weight.detach()
        )
        script_model.sub.fc2.bias = torch.nn.Parameter(
            eager_model.sub.module.fc2.bias.detach()
        )
        script_model.fc.weight = torch.nn.Parameter(eager_model.fc.weight.detach())
        script_model.fc.bias = torch.nn.Parameter(eager_model.fc.bias.detach())

        eager_model.fuse_modules()

        model_eager = quantize(eager_model, test_only_eval_fn, [self.calib_data])
        qconfig_dict = {
            "": get_default_qconfig(torch.backends.quantized.engine),
            "fc": None,
        }
        model_traced = torch.jit.trace(script_model, self.calib_data[0][0])
        model_script = torch.jit.script(script_model)
        result_eager = model_eager(self.calib_data[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(
                model_under_test,
                qconfig_dict,
                test_only_eval_fn,
                [self.calib_data],
                inplace=False,
            )
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines
    def test_single_linear_dynamic(self):
        # [FUNCTION DOC] Compares eager and graph mode dynamic quantization of a single linear.
        # Only runs when the engine is qnnpack. For traced and scripted models requires
        # quantize_dynamic_jit output to match eager dynamic quantization, and that the debug
        # variant (fake-quantized with choose-qparams, quantize, dequantize, linear) also matches.
        r"""Compare the result of dynamic quantization of single linear layer in
        eager mode and graph mode.
        """
        if qengine_is_qnnpack():
            # eager mode
            annotated_linear_model = AnnotatedSingleLayerLinearModel("qnnpack").eval()
            linear_model = SingleLayerLinearModel().eval()
            # copy the weight from eager mode so that we can
            # compare the result of the two quantized models later
            linear_model.fc1.weight = torch.nn.Parameter(
                annotated_linear_model.fc1.module.weight.detach()
            )
            linear_model.fc1.bias = torch.nn.Parameter(
                annotated_linear_model.fc1.module.bias.detach()
            )
            qconfig_dict = {"": default_dynamic_qconfig}
            model_eager = quantize_dynamic(annotated_linear_model, qconfig_dict)

            model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
            model_script = torch.jit.script(linear_model)
            result_eager = model_eager(self.calib_data[0][0])

            for model_under_test in [model_traced, model_script]:
                model_quantized = quantize_dynamic_jit(model_under_test, qconfig_dict)
                self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

                # Check to make sure choose_qparams->quant->dequant->linear is numerically
                # equivalent to the final quantized model.
                model_fake_quantized = quantize_dynamic_jit(
                    model_under_test, qconfig_dict, debug=True
                )
                self.assertEqual(
                    model_fake_quantized(self.calib_data[0][0]), result_eager
                )

    @skipIfNoFBGEMM
    def test_linear_dynamic_fp16(self):
        # [FUNCTION DOC] Checks fp16 dynamic quantization of a linear whose weights (65532) exceed
        # fp16 range: eager quantize_dynamic with float16 gives a reference; the traced model
        # checked with the graph mode helper expecting the dynamic fp16 linear op, with warnings
        # captured and not asserted on, must produce an equal output. Skipped without fbgemm.
        linear_model = SingleLayerLinearModel().eval()
        # Create weight tensor values that are beyond fp16 max
        x = torch.ones(5, 5) * 65532
        linear_model.fc1.weight = torch.nn.Parameter(x)
        import warnings

        model_eager = quantize_dynamic(linear_model, dtype=torch.float16)
        result_eager = model_eager(self.calib_data[0][0])
        for trace in [True]:
            with warnings.catch_warnings(record=True) as w:
                quantized_model = self.checkGraphModeOp(
                    linear_model,
                    self.calib_data[0][0],
                    "quantized::linear_dynamic_fp16",
                    tracing=trace,
                    dynamic=True,
                    qconfig=float16_dynamic_qconfig,
                )
            # compare result with eager mode
            self.assertEqual(quantized_model(self.calib_data[0][0]), result_eager)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
