# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Backward-compatibility tests for quantization: load artifacts saved by older
# releases from the serialized directory next to the test directory (inputs, state dicts,
# scripted and traced modules, expected outputs, torch.package archives) and check current code
# reproduces the saved results. Covers quantized linear, conv, dynamic linear and LSTM modules,
# graph-mode quantized models, observers, a QAT-qconfig model and an FX-quantization transform
# of a packaged model. All tests run with generate off; the generate paths only write new
# artifacts when the fbgemm engine is active. Direct execution is refused via
# raise_on_run_directly.

import os
import sys
import unittest

# torch
import torch
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn as nn
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver
from torch.fx import GraphModule
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.common_quantized import (
    override_qengines,
    qengine_is_fbgemm,
)

# Testing utils
from torch.testing._internal.common_utils import (
    IS_AVX512_VNNI_SUPPORTED,
    raise_on_run_directly,
    TestCase,
)
from torch.testing._internal.quantization_torch_package_models import (
    LinearReluFunctional,
)


def remove_prefix(text, prefix):
    # [FUNCTION DOC] HELPER: Returns the text without the given prefix when it starts with that
    # prefix, otherwise returns the text unchanged.
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_filenames(self, subname):
    # [FUNCTION DOC] HELPER: Builds the artifact file names for the running test. The base name
    # is the test id with the module prefix removed, placed in the serialized directory one
    # level above the directory of the test module's file, with an underscore plus subname
    # appended when a subname is given. Returns a tuple of seven paths in order: input, state
    # dict, scripted module, traced module, expected output, package and get-attr targets files.
    # Only computes strings; touches no files.
    # NB: we take __file__ from the module that defined the test
    # class, so we place the expect directory where the test script
    # lives, NOT where test/common_utils.py lives.
    module_id = self.__class__.__module__
    munged_id = remove_prefix(self.id(), module_id + ".")
    test_file = os.path.realpath(sys.modules[module_id].__file__)
    base_name = os.path.join(os.path.dirname(test_file), "../serialized", munged_id)

    if subname:
        base_name += "_" + subname

    input_file = base_name + ".input.pt"
    state_dict_file = base_name + ".state_dict.pt"
    scripted_module_file = base_name + ".scripted.pt"
    traced_module_file = base_name + ".traced.pt"
    expected_file = base_name + ".expected.pt"
    package_file = base_name + ".package.pt"
    get_attr_targets_file = base_name + ".get_attr_targets.pt"

    return (
        input_file,
        state_dict_file,
        scripted_module_file,
        traced_module_file,
        expected_file,
        package_file,
        get_attr_targets_file,
    )


class TestSerialization(TestCase):
    # [CLASS DOC] Test case holding the helpers that load saved artifacts and compare them with
    # current behavior, plus the per-module backward-compatibility tests. Most tests use the
    # override-qengines decorator so they run for each available quantization engine. Artifact
    # names are derived from the test name, so each test needs its own files.
    """Test backward compatibility for serialization and numerics"""

    # Copy and modified from TestCase.assertExpected
    def _test_op(
        self,
        qmodule,
        subname=None,
        input_size=None,
        input_quantized=True,
        generate=False,
        prec=None,
        new_zipfile_serialization=False,
    ):
        # [FUNCTION DOC] HELPER: Checks a quantized module can load an old serialized state. If
        # generate is set and the fbgemm engine is active, first writes new artifacts: a random
        # float input (quantized to quint8 with scale 0.5 and zero point 2 when
        # input_quantized), the state dict (with the zipfile flag controlled by
        # new_zipfile_serialization), scripted and traced modules, and the expected output. Then
        # loads the input, loads the state dict into the given module (weights_only off), loads
        # the scripted and traced modules, and asserts all three outputs equal the saved
        # expected output within atol prec.
        r"""Test quantized modules serialized previously can be loaded
        with current code, make sure we don't break backward compatibility for the
        serialization of quantized modules
        """
        (
            input_file,
            state_dict_file,
            scripted_module_file,
            traced_module_file,
            expected_file,
            _package_file,
            _get_attr_targets_file,
        ) = get_filenames(self, subname)

        # only generate once.
        if generate and qengine_is_fbgemm():
            input_tensor = torch.rand(*input_size).float()
            if input_quantized:
                input_tensor = torch.quantize_per_tensor(
                    input_tensor, 0.5, 2, torch.quint8
                )
            torch.save(input_tensor, input_file)
            # Temporary fix to use _use_new_zipfile_serialization until #38379 lands.
            torch.save(
                qmodule.state_dict(),
                state_dict_file,
                _use_new_zipfile_serialization=new_zipfile_serialization,
            )
            torch.jit.save(torch.jit.script(qmodule), scripted_module_file)
            torch.jit.save(torch.jit.trace(qmodule, input_tensor), traced_module_file)
            torch.save(qmodule(input_tensor), expected_file)

        input_tensor = torch.load(input_file)
        # weights_only = False as sometimes get ScriptObject here
        qmodule.load_state_dict(torch.load(state_dict_file, weights_only=False))
        qmodule_scripted = torch.jit.load(scripted_module_file)
        qmodule_traced = torch.jit.load(traced_module_file)
        expected = torch.load(expected_file)
        self.assertEqual(qmodule(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_scripted(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_traced(input_tensor), expected, atol=prec)

    def _test_op_graph(
        self,
        qmodule,
        subname=None,
        input_size=None,
        input_quantized=True,
        generate=False,
        prec=None,
        new_zipfile_serialization=False,
    ):
        # [FUNCTION DOC] HELPER: Graph-mode variant for a float module. If generate is set and
        # fbgemm is active, writes a random input, scripts and traces the module, quantizes both
        # with the jit quantization routine using the default qconfig and a simple evaluation
        # function, and saves both plus the scripted output as expected. Always then loads the
        # saved input, scripted and traced models and expected output and asserts both models
        # reproduce the expected output within atol prec. The module itself is only used when
        # generating; the state dict file is not used.
        r"""
        Input: a floating point module

        If generate == True, traces and scripts the module and quantizes the results with
        PTQ, and saves the results.

        If generate == False, traces and scripts the module and quantizes the results with
        PTQ, and compares to saved results.
        """
        (
            input_file,
            _,
            scripted_module_file,
            traced_module_file,
            expected_file,
            _package_file,
            _get_attr_targets_file,
        ) = get_filenames(self, subname)

        # only generate once.
        if generate and qengine_is_fbgemm():
            input_tensor = torch.rand(*input_size).float()
            torch.save(input_tensor, input_file)

            # convert to TorchScript
            scripted = torch.jit.script(qmodule)
            traced = torch.jit.trace(qmodule, input_tensor)

            # quantize

            def _eval_fn(model, data):
                # [FUNCTION DOC] HELPER: Calibration callback for quantize_jit: runs the model
                # once on the data for its side effect (observer statistics); returns nothing.
                model(data)

            qconfig_dict = {"": torch.ao.quantization.default_qconfig}
            scripted_q = torch.ao.quantization.quantize_jit(
                scripted, qconfig_dict, _eval_fn, [input_tensor]
            )
            traced_q = torch.ao.quantization.quantize_jit(
                traced, qconfig_dict, _eval_fn, [input_tensor]
            )

            torch.jit.save(scripted_q, scripted_module_file)
            torch.jit.save(traced_q, traced_module_file)
            torch.save(scripted_q(input_tensor), expected_file)

        input_tensor = torch.load(input_file)
        qmodule_scripted = torch.jit.load(scripted_module_file)
        qmodule_traced = torch.jit.load(traced_module_file)
        expected = torch.load(expected_file)
        self.assertEqual(qmodule_scripted(input_tensor), expected, atol=prec)
        self.assertEqual(qmodule_traced(input_tensor), expected, atol=prec)

    def _test_obs(
        self, obs, input_size, subname=None, generate=False, check_numerics=True
    ):
        # [FUNCTION DOC] HELPER: Checks an observer or observed module can load an old state
        # dict. If generate is set, saves a random float input, the observer's output on it and
        # its state dict. Always loads the input and state dict into the given object; when
        # check_numerics is true also asserts the observer's output on the input equals the
        # saved expected output.
        """
        Test observer code can be loaded from state_dict.
        """
        (
            input_file,
            state_dict_file,
            _,
            _,
            expected_file,
            _package_file,
            _get_attr_targets_file,
        ) = get_filenames(self, None)
        if generate:
            input_tensor = torch.rand(*input_size).float()
            torch.save(input_tensor, input_file)
            torch.save(obs(input_tensor), expected_file)
            torch.save(obs.state_dict(), state_dict_file)

        input_tensor = torch.load(input_file)
        obs.load_state_dict(torch.load(state_dict_file))
        expected = torch.load(expected_file)
        if check_numerics:
            self.assertEqual(obs(input_tensor), expected)

    def _test_package(self, fp32_module, input_size, generate=False):
        # [FUNCTION DOC] HELPER: Checks old torch.package archives still work with current FX
        # graph-mode quantization. If generate is set and fbgemm is active, saves a random
        # input, exports the float module into a package archive, applies the FX transforms and
        # saves the set of get_attr target names and the quantized output. Always loads the
        # saved files, imports the model from the archive, re-applies the transforms, asserts
        # the get_attr target names equal the saved set and that the output is allclose to the
        # saved output.
        """
        Verifies that files created in the past with torch.package
        work on today's FX graph mode quantization transforms.
        """
        (
            input_file,
            _,
            _scripted_module_file,
            _traced_module_file,
            expected_file,
            package_file,
            get_attr_targets_file,
        ) = get_filenames(self, None)

        package_name = "test"
        resource_name_model = "test.pkl"

        def _do_quant_transforms(
            m: torch.nn.Module,
            input_tensor: torch.Tensor,
        ) -> torch.nn.Module:
            # [FUNCTION DOC] HELPER: Applies FX post-training static quantization to the module
            # using the fbgemm default qconfig for all modules: prepares with the input as
            # example input, runs one calibration pass, converts and returns the quantized
            # module.
            example_inputs = (input_tensor,)
            # do the quantizaton transforms and save result
            qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
            mp = quantize_fx.prepare_fx(m, {"": qconfig}, example_inputs=example_inputs)
            mp(input_tensor)
            mq = quantize_fx.convert_fx(mp)
            return mq

        def _get_get_attr_target_strings(m: GraphModule) -> set[str]:
            # [FUNCTION DOC] HELPER: Returns the set of target names of all attribute-fetch
            # nodes in the graph of the given graph module.
            results = set()
            for node in m.graph.nodes:
                if node.op == "get_attr":
                    results.add(node.target)
            return results

        if generate and qengine_is_fbgemm():
            input_tensor = torch.randn(*input_size)
            torch.save(input_tensor, input_file)

            # save the model with torch.package
            with torch.package.PackageExporter(package_file) as exp:
                exp.intern("torch.testing._internal.quantization_torch_package_models")
                exp.save_pickle(package_name, resource_name_model, fp32_module)

            # do the quantization transforms and save the result
            mq = _do_quant_transforms(fp32_module, input_tensor)
            get_attrs = _get_get_attr_target_strings(mq)
            torch.save(get_attrs, get_attr_targets_file)
            q_result = mq(input_tensor)
            torch.save(q_result, expected_file)

        # load input tensor
        input_tensor = torch.load(input_file)
        expected_output_tensor = torch.load(expected_file)
        expected_get_attrs = torch.load(get_attr_targets_file, weights_only=False)

        # load model from package and verify output and get_attr targets match
        imp = torch.package.PackageImporter(package_file)
        m = imp.load_pickle(package_name, resource_name_model)
        mq = _do_quant_transforms(m, input_tensor)

        get_attrs = _get_get_attr_target_strings(mq)
        self.assertTrue(
            get_attrs == expected_get_attrs,
            lambda msg: f"{msg}\nget_attrs: expected {expected_get_attrs}, got {get_attrs}",
        )
        output_tensor = mq(input_tensor)
        self.assertTrue(torch.allclose(output_tensor, expected_output_tensor))

    @override_qengines
    def test_linear(self):
        # [FUNCTION DOC] Loads saved artifacts for a quantized linear module with 3 inputs, 1
        # output, bias and 8-bit signed weights; input shape 1 by 3.
        module = nnq.Linear(3, 1, bias_=True, dtype=torch.qint8)
        self._test_op(module, input_size=[1, 3], generate=False)

    @override_qengines
    def test_linear_relu(self):
        # [FUNCTION DOC] Loads saved artifacts for the fused quantized linear-ReLU module (3 in,
        # 1 out, bias, 8-bit signed); input shape 1 by 3.
        module = nniq.LinearReLU(3, 1, bias=True, dtype=torch.qint8)
        self._test_op(module, input_size=[1, 3], generate=False)

    @override_qengines
    def test_linear_dynamic(self):
        # [FUNCTION DOC] Dynamic quantized linear with a float input: checks the qint8 variant
        # (artifact subname qint8) and, only on the fbgemm engine, the float16 variant (subname
        # float16).
        module_qint8 = nnqd.Linear(3, 1, bias_=True, dtype=torch.qint8)
        self._test_op(
            module_qint8,
            "qint8",
            input_size=[1, 3],
            input_quantized=False,
            generate=False,
        )
        if qengine_is_fbgemm():
            module_float16 = nnqd.Linear(3, 1, bias_=True, dtype=torch.float16)
            self._test_op(
                module_float16,
                "float16",
                input_size=[1, 3],
                input_quantized=False,
                generate=False,
            )

    @override_qengines
    def test_conv2d(self):
        # [FUNCTION DOC] Loads saved artifacts for a quantized 2D convolution (3 to 3 channels,
        # kernel 3, with bias); input shape 1 by 3 by 6 by 6.
        module = nnq.Conv2d(
            3,
            3,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_nobias(self):
        # [FUNCTION DOC] Same as the 2D convolution test but built without a bias.
        module = nnq.Conv2d(
            3,
            3,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_graph(self):
        # [FUNCTION DOC] Graph-mode test: a float sequential of a quant stub and a 2D
        # convolution with bias is compared against saved scripted and traced quantized models.
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_nobias_graph(self):
        # [FUNCTION DOC] Same as the graph-mode convolution test but the convolution has no
        # bias.
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_graph_v2(self):
        # [FUNCTION DOC] Same body as the graph-mode convolution test; uses its own saved
        # artifacts for version 2 of the conv packed parameters.
        # tests the same thing as test_conv2d_graph, but for version 2 of
        # ConvPackedParams{n}d
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_nobias_graph_v2(self):
        # [FUNCTION DOC] Same body as the no-bias graph-mode convolution test; own artifacts for
        # version 2 of the conv packed parameters.
        # tests the same thing as test_conv2d_nobias_graph, but for version 2 of
        # ConvPackedParams{n}d
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_graph_v3(self):
        # [FUNCTION DOC] Same body as the graph-mode convolution test; own artifacts for version
        # 3 of the conv packed parameters.
        # tests the same thing as test_conv2d_graph, but for version 3 of
        # ConvPackedParams{n}d
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_nobias_graph_v3(self):
        # [FUNCTION DOC] Same body as the no-bias graph-mode convolution test; own artifacts for
        # version 3 of the conv packed parameters.
        # tests the same thing as test_conv2d_nobias_graph, but for version 3 of
        # ConvPackedParams{n}d
        module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                padding_mode="zeros",
            ),
        )
        self._test_op_graph(module, input_size=[1, 3, 6, 6], generate=False)

    @override_qengines
    def test_conv2d_relu(self):
        # [FUNCTION DOC] Loads saved artifacts for the fused quantized 2D conv-ReLU module with
        # bias; input shape 1 by 3 by 6 by 6.
        module = nniq.ConvReLU2d(
            3,
            3,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self._test_op(module, input_size=[1, 3, 6, 6], generate=False)
        # TODO: graph mode quantized conv2d module

    @override_qengines
    def test_conv3d(self):
        # [FUNCTION DOC] Only on the fbgemm engine: loads saved artifacts for a quantized 3D
        # convolution with bias; input shape 1 by 3 by 6 by 6 by 6. Does nothing on other
        # engines.
        if qengine_is_fbgemm():
            module = nnq.Conv3d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            )
            self._test_op(module, input_size=[1, 3, 6, 6, 6], generate=False)
            # TODO: graph mode quantized conv3d module

    @override_qengines
    def test_conv3d_relu(self):
        # [FUNCTION DOC] Only on the fbgemm engine: loads saved artifacts for the fused
        # quantized 3D conv-ReLU module with bias. Does nothing on other engines.
        if qengine_is_fbgemm():
            module = nniq.ConvReLU3d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            )
            self._test_op(module, input_size=[1, 3, 6, 6, 6], generate=False)
            # TODO: graph mode quantized conv3d module

    @override_qengines
    @unittest.skipIf(
        IS_AVX512_VNNI_SUPPORTED,
        "This test fails on machines with AVX512_VNNI support. Ref: GH Issue 59098",
    )
    def test_lstm(self):
        # [FUNCTION DOC] Only on the fbgemm engine: wraps a dynamic quantized LSTM (input 3,
        # hidden 7, one layer, converted to float dtype) in a small module and checks it against
        # saved artifacts with non-quantized input shape 4 by 4 by 3 and the new zipfile flag.
        # Skipped on machines with AVX512 VNNI support because of a known failure (GitHub issue
        # 59098).
        class LSTMModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Local wrapper module holding one dynamic quantized LSTM
            # attribute; its forward returns the LSTM's result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dynamic quantized LSTM (input size 3,
                # hidden size 7, one layer) and converts it to float dtype.
                super().__init__()
                self.lstm = nnqd.LSTM(input_size=3, hidden_size=7, num_layers=1).to(
                    dtype=torch.float
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the LSTM on the input and returns whatever it
                # returns (output and hidden-state tuple).
                x = self.lstm(x)
                return x

        if qengine_is_fbgemm():
            mod = LSTMModule()
            self._test_op(
                mod,
                input_size=[4, 4, 3],
                input_quantized=False,
                generate=False,
                new_zipfile_serialization=True,
            )

    def test_per_channel_observer(self):
        # [FUNCTION DOC] Loads a saved per-channel min-max observer state and checks its output
        # on a 5 by 5 input equals the saved one.
        obs = PerChannelMinMaxObserver()
        self._test_obs(obs, input_size=[5, 5], generate=False)

    def test_per_tensor_observer(self):
        # [FUNCTION DOC] Same as the per-channel test but for the per-tensor min-max observer.
        obs = MinMaxObserver()
        self._test_obs(obs, input_size=[5, 5], generate=False)

    def test_default_qat_qconfig(self):
        # [FUNCTION DOC] Builds a linear plus ReLU model with random weights and the default
        # fbgemm QAT qconfig, wraps it with a quant wrapper and prepares it for QAT, then checks
        # the saved state dict can be loaded into it; numerics are not checked.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Local float model of a 5 by 5 linear layer followed by ReLU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 5 by 5 linear layer and the ReLU
                # submodules.
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer then ReLU and returns the
                # result.
                x = self.linear(x)
                x = self.relu(x)
                return x

        model = Model()
        model.linear.weight = torch.nn.Parameter(torch.randn(5, 5))
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
        ref_model = torch.ao.quantization.QuantWrapper(model)
        ref_model = torch.ao.quantization.prepare_qat(ref_model)
        self._test_obs(
            ref_model, input_size=[5, 5], generate=False, check_numerics=False
        )

    @skipIfNoFBGEMM
    def test_linear_relu_package_quantization_transforms(self):
        # [FUNCTION DOC] Skipped without fbgemm. Builds the packaged linear-ReLU functional
        # model in eval mode (4 features) and checks old torch.package archives still give the
        # saved FX-quantized output and get_attr targets with input shape 1 by 1 by 4 by 4.
        m = LinearReluFunctional(4).eval()
        self._test_package(m, input_size=(1, 1, 4, 4), generate=False)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
