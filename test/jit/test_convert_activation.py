# Owner(s): ["oncall: jit"]

# [MODULE DOC] Tests for the TorchScript graph passes that convert between functional and
# in-place activation operators. Defines a module-level list of eleven torch.nn.functional
# activations and two JitTestCase classes, one per pass. An optional torchvision import
# sets a flag that gates two resnet18 tests. Running the file directly is rejected.
import os
import sys
import unittest
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck


try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


activations = [
    F.celu,
    F.elu,
    F.hardsigmoid,
    F.hardswish,
    F.hardtanh,
    F.leaky_relu,
    F.relu,
    F.relu6,
    F.rrelu,
    F.selu,
    F.silu,
]


class TestFunctionalToInplaceActivation(JitTestCase):
    # [CLASS DOC] Tests for the functional_to_inplace_activation pass, which rewrites eligible
    # functional activations in a scripted graph to their in-place variants.
    def test_check_no_type_promotion(self):
        # [FUNCTION DOC] For every activation and every dtype from bool through float64, apply
        # the activation to a random tensor of that dtype and require the result dtype to be
        # unchanged. Operators that raise RuntimeError for a dtype are skipped. Guards the
        # type-conversion assumption encoded in the C++ restore-mutation table.
        dtypes = [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
        ]
        # restore_mutation.h contains a mapping from activation operators
        # to whether they allow type conversion. Use this checking to
        # guard the mapping, and if any later change breaks the assumption
        # we need to update the mapping correspondingly.
        for activation, dtype in product(activations, dtypes):
            inp = torch.normal(0, 5, size=(4, 4)).to(dtype)
            try:
                out = activation(inp)
                self.assertEqual(dtype, out.dtype)
            except RuntimeError:
                # Skip the not implemented error
                pass

    def test_functional_to_inplace_activation(self):
        # [FUNCTION DOC] For each activation, script a small function that adds one then
        # applies it. After inlining and constant propagation the graph must contain the
        # functional operator; after the pass it must contain only the in-place operator.
        # Finally the scripted result must equal the eager result on a random input.
        for activation in activations:

            def test_basic(x):
                # [FUNCTION DOC] HELPER: Add one to the input and return the
                # activation applied to the result; the activation comes from the
                # enclosing loop variable.
                y = x + 1
                z = activation(y)
                return z

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)
            self.run_pass("functional_to_inplace_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}(").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            inp = torch.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

    def test_no_functional_to_inplace(self):
        # [FUNCTION DOC] Three negative cases where the pass must not produce in-place ops:
        # sigmoid (may change dtype), relu on a view of the input (aliasing), and relu on a
        # module attribute (global scope). Each is scripted, the pass is run, and the graph is
        # checked to lack the in-place operator.
        # inplace conversion should not happen because sigmoid may
        # perform type conversion
        def test1():
            # [FUNCTION DOC] HELPER: Build a ones tensor and return its sigmoid.
            y = torch.ones([2, 2])
            z = torch.sigmoid(y)
            return z

        fn = torch.jit.script(test1)
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::sigmoid_").run(fn.graph)

        # inplace conversion should not happen because y is alias
        # the input x
        def test2(x):
            # [FUNCTION DOC] HELPER: Take the first element of the input and return its relu.
            y = x[0]
            z = torch.relu(y)
            return z

        fn = torch.jit.script(test2)
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::relu_").run(fn.graph)

        # inplace conversion should not happen because self.x is
        # at the global scope
        class Test3(nn.Module):
            # [CLASS DOC] HELPER: Module storing a tensor attribute whose forward returns relu
            # of it.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the given tensor as an attribute.
                super().__init__()
                self.x = x

            def forward(self):
                # [FUNCTION DOC] HELPER: Return relu of the stored attribute.
                y = torch.relu(self.x)
                return y

        fn = torch.jit.script(Test3(torch.rand([2, 2])).eval())
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::relu_").run(fn.graph)

    @skipIfNoTorchVision
    def test_resnet18_correctness(self):
        # [FUNCTION DOC] Skipped without torchvision. Script and freeze a resnet18 in eval
        # mode, run the pass on the frozen graph, and require its output on a random batch of
        # ten 3x224x224 images to equal the eager model output.
        model = torchvision.models.resnet18()
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        inp = torch.randn(N, C, H, W)
        self.run_pass("functional_to_inplace_activation", frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))


class TestInplaceToFunctionalActivation(JitTestCase):
    # [CLASS DOC] Tests for the inplace_to_functional_activation pass, which rewrites in-place
    # activations in a scripted graph back to functional operators.
    def test_inplace_to_functional_activation(self):
        # [FUNCTION DOC] Two loops. First, for each activation called with the inplace flag,
        # check the graph holds the in-place operator, run the pass, then require the
        # functional operator and no in-place one. Second, for relu_, sigmoid_ and tanh_
        # called directly, do the same comparison by operator name and also require scripted
        # and eager outputs to match.
        for activation in activations:

            def test_basic(x):
                # [FUNCTION DOC] HELPER: Add one to the input, apply the activation in place
                # with the inplace flag, and return the modified tensor.
                y = x + 1
                activation(y, inplace=True)
                return y

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            self.run_pass("inplace_to_functional_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}_").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)

        for activation in [
            torch.relu_,
            torch.sigmoid_,
            torch.tanh_,
        ]:

            def test_basic(x):
                # [FUNCTION DOC] HELPER: Add one to the input, apply the in-place activation
                # function to it, and return the modified tensor.
                y = x + 1
                activation(y)
                return y

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}").run(fn.graph)
            self.run_pass("inplace_to_functional_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__[:-1]}(").run(fn.graph)

            inp = torch.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

    @skipIfNoTorchVision
    def test_resnet18_correctness(self):
        # [FUNCTION DOC] Skipped without torchvision. Script and freeze a resnet18 in eval
        # mode, run the inplace-to-functional pass on the frozen graph, and require its output
        # on a random batch to equal the eager model output.
        model = torchvision.models.resnet18()
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        inp = torch.randn(N, C, H, W)
        self.run_pass("inplace_to_functional_activation", frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
