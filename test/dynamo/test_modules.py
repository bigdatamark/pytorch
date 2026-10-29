# Owner(s): ["module: dynamo"]
# ruff: noqa: F841
# [MODULE DOC] Tests for how Dynamo traces and compiles torch.nn.Module instances. The top part
# defines many small module classes (each exercising one tracing feature such as submodule
# calls, static or class methods, properties, module containers, lazy modules, parameter
# iteration, super calls and attribute precedence) plus helpers; later test classes compile them
# and compare against eager. Importing has no side effects beyond defining two module-level
# counters and importing the sibling test_functions module (falling back to a plain import when
# run as a script). At the end it registers device-specific variants of NNModuleTestsDevice for
# non-CPU devices (allowing XPU) and runs the Dynamo test runner when executed as a script.

import collections
import copy
import itertools
import os
import tempfile
import traceback
import types
import unittest
from copy import deepcopy
from functools import partial, update_wrapper
from typing import NamedTuple
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.nn.functional as F
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.eval_frame import unsupported
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.testing import same
from torch._dynamo.utils import ifdynstaticdefault
from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfHpu


try:
    from . import test_functions
except ImportError:
    import test_functions


_variable = 0
_variable1 = 0


def update_global():
    # [FUNCTION DOC] HELPER: Increments both module-level counters _variable and _variable1 by
    # one using global declarations; used as a function with global side effects.
    global _variable, _variable1
    _variable += 1
    _variable1 += 1


class BasicModule(torch.nn.Module):
    # [CLASS DOC] Simple module with one Linear layer and a fixed random scale tensor attribute;
    # baseline module used by many other test modules and by make_test.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a Linear layer of width 10 and stores a random one-by-
        # ten tensor as a plain attribute named scale.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies linear1, relu, then multiplies by scale; returns the
        # resulting tensor.
        return F.relu(self.linear1(x)) * self.scale


class FnMember(torch.nn.Module):
    # [CLASS DOC] Module holding a Linear layer and a plain function (relu) stored as an
    # attribute, to test calling function-valued members.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1 and stores the functional relu as the
        # activation attribute.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = F.relu

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs linear1, then applies the activation attribute if it is
        # truthy; returns the result.
        x = self.linear1(x)
        if self.activation:
            x = self.activation(x)
        return x


class FnMemberCmp(torch.nn.Module):
    # [CLASS DOC] Module whose activation is a constructor argument that may be None, to test
    # identity comparisons against None on members.
    def __init__(self, activation):
        # [FUNCTION DOC] HELPER: Creates linear1 and stores the given activation (a callable or
        # None) as an attribute.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = activation

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs linear1; applies the activation when it is not None, and
        # applies sigmoid when it is None; returns the result.
        x = self.linear1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.activation is None:
            x = torch.sigmoid(x)
        return x


class SubmoduleExample(torch.nn.Module):
    # [CLASS DOC] Module composed of two BasicModule submodules and a fixed random scale tensor.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1 and layer2 as BasicModule instances and a random
        # one-by-ten scale tensor.
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs x through layer1 then layer2 and multiplies the result by
        # scale.
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(torch.nn.Module):
    # [CLASS DOC] Module that selects one of two Linear layers depending on its training flag;
    # the constructor puts it in training mode.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1 and linear2 and calls train with True.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.train(True)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Chooses linear1 when the module is in training mode, otherwise
        # linear2, and returns relu of that layer applied to x.
        if self.training:
            mod = self.linear1
        else:
            mod = self.linear2
        return F.relu(mod(x))


class IsEvalCheck(IsTrainingCheck):
    # [CLASS DOC] Subclass of IsTrainingCheck that is switched to evaluation mode after parent
    # construction.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the parent constructor, then calls train with False so
        # linear2 is used in forward.
        super().__init__()
        self.train(False)


class ModuleMethodCall(torch.nn.Module):
    # [CLASS DOC] Module that calls its submodules through a helper method on itself rather than
    # directly.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1 and layer2 as BasicModule instances and a random
        # scale tensor.
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        # [FUNCTION DOC] HELPER: Calls the given module on x and multiplies the output by
        # self.scale; returns the product.
        x = mod(x)
        return x * self.scale

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies call_and_scale with layer1 and with layer2 on the same
        # input and returns the sum of both results.
        x1 = self.call_and_scale(self.layer1, x)
        x2 = self.call_and_scale(self.layer2, x)
        return x1 + x2


class UnsupportedMethodCall(torch.nn.Module):
    # [CLASS DOC] Module whose helper method ends in a call to the Dynamo-unsupported function,
    # forcing a graph break inside a method.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1 (BasicModule) and a random scale tensor.
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        # [FUNCTION DOC] HELPER: Calls the module, scales the output, then passes it to the
        # unsupported function (which Dynamo cannot trace) and returns that function's result.
        x = mod(x)
        x = x * self.scale
        return unsupported(x, x)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds the input to the result of call_and_scale on layer1 and
        # returns it.
        x1 = self.call_and_scale(self.layer1, x)
        return x + x1


class UnsupportedModule(torch.nn.Module):
    # [CLASS DOC] Module whose forward ends in a call to the Dynamo-unsupported function.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1 (BasicModule) and a random scale tensor.
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs layer1, scales the result, then returns the result of
        # calling the unsupported function on it.
        x = self.layer1(x) * self.scale
        return unsupported(x, x)


class UnsupportedModuleCall(torch.nn.Module):
    # [CLASS DOC] Wrapper that calls an UnsupportedModule submodule from forward to test graph
    # breaks across module boundaries.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates an UnsupportedModule instance stored as mod.
        super().__init__()
        self.mod = UnsupportedModule()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Scales the input by 1.5, passes it through mod, and returns one
        # plus the result.
        return 1 + self.mod(x * 1.5)


class ModuleWithStaticForward(torch.nn.Module):
    # [CLASS DOC] Module whose forward is declared as a static method, so it takes only the
    # input tensor.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Static forward; returns x multiplied by sigmoid of x.
        return x * torch.sigmoid(x)


class ModuleCallModuleWithStaticForward(torch.nn.Module):
    # [CLASS DOC] Wrapper that holds a ModuleWithStaticForward submodule and calls it.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a ModuleWithStaticForward submodule named mod.
        super().__init__()
        self.mod = ModuleWithStaticForward()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the output of mod applied to x.
        return self.mod(x)


class ModuleStaticMethodCall(torch.nn.Module):
    # [CLASS DOC] Module that calls a static helper method (taking scale, module, and input) on
    # itself from forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1 and layer2 BasicModule instances and a random
        # scale tensor.
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @staticmethod
    def call_and_scale(scale, mod, x):
        # [FUNCTION DOC] HELPER: Static helper; calls the given module on x and multiplies the
        # result by the given scale.
        x = mod(x)
        return x * scale

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls the static helper through self for layer1 and layer2 and
        # returns the sum of both outputs.
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleClassMethodCall(torch.nn.Module):
    # [CLASS DOC] Module that calls a class method helper from forward; same shape as
    # ModuleStaticMethodCall but with classmethod.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1 and layer2 BasicModule instances and a random
        # scale tensor.
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @classmethod
    def call_and_scale(cls, scale, mod, x):
        # [FUNCTION DOC] HELPER: Class method helper; calls the given module on x and multiplies
        # by the given scale. The class argument is unused.
        x = mod(x)
        return x * scale

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls the class method helper with layer1 and layer2 and
        # returns the sum of the two results.
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleProperty(torch.nn.Module):
    # [CLASS DOC] Module exposing its scale tensor through a property, used in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a random one-by-ten scale tensor.
        super().__init__()
        self.scale = torch.randn(1, 10)

    @property
    def scale_alias(self):
        # [FUNCTION DOC] HELPER: Property getter returning the scale attribute.
        return self.scale

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Multiplies x by the scale_alias property.
        return x * self.scale_alias


class NestedModuleList(torch.nn.Module):
    # [CLASS DOC] Module holding a ModuleList of three inner ModuleLists, each containing a
    # Linear and a ReLU, iterated with tuple unpacking.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds an outer ModuleList and appends three inner ModuleLists
        # of one Linear of size ten and one ReLU each.
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(3):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.Linear(10, 10),
                        torch.nn.ReLU(),
                    ]
                )
            )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Iterates the outer list, unpacking each inner pair into a layer
        # and an activation, and applies activation of layer of x in turn; returns the result.
        for layer, act in self.layers:
            x = act(layer(x))
        return x


class ConstLoop(torch.nn.Module):
    # [CLASS DOC] Module with a Linear layer applied repeatedly in a loop whose trip count is a
    # plain integer attribute.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1 and sets count to three.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.count = 3

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies sigmoid of linear1 to x count times in a loop and
        # returns the result.
        for _ in range(self.count):
            x = torch.sigmoid(self.linear1(x))
        return x


class ViaModuleCall(torch.nn.Module):
    # [CLASS DOC] Module that calls a helper from test_functions (constant3) from forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls the test_functions helper constant3 with the sigmoid of
        # linear1 of x and with x; returns its result.
        return test_functions.constant3(torch.sigmoid(self.linear1(x)), x)


class IsNoneLayer(torch.nn.Module):
    # [CLASS DOC] Module with an optional layer: layer1 is a Linear and layer2 is None, and
    # forward skips layers that are None.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1 as Linear, sets layer2 to None, and puts the
        # module in training mode.
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = None
        self.train(True)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies layer1 and layer2 in order, skipping any that is None;
        # returns the result.
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        return x


class LayerList(torch.nn.Module):
    # [CLASS DOC] Module storing a plain Python list of layers (not a ModuleList) and iterating
    # it.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Stores a list of Linear, ReLU, Linear as attribute layers.
        super().__init__()
        self.layers = [
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
        ]

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies each layer in the list in order and returns the result.
        for layer in self.layers:
            x = layer(x)
        return x


class ModuleList(torch.nn.Module):
    # [CLASS DOC] Module exercising many ModuleList iteration patterns: indexing by range,
    # direct iteration, zip with tensors and with ints, enumerate, and enumerate over a reversed
    # slice.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a ModuleList of Linear, ReLU, Linear, ReLU.
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs six passes over the layers (index loop, plain loop, zip
        # with a tuple of tensors, zip with integers, enumerate multiplying by the index,
        # enumerate over the reversed list multiplying by the index), feeding each result
        # forward; returns the final tensor.
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        for layer in self.layers:
            x = layer(x)

        for layer, val in zip(self.layers, (x, x, x, x)):
            x = layer(x) + val

        for layer, val in zip(self.layers, (1, 2, 3, 4)):
            x = layer(x) + val

        for idx, layer in enumerate(self.layers):
            x = layer(x) * idx

        for idx, layer in enumerate(self.layers[::-1]):
            x = layer(x) * idx

        return x


class CustomGetItemModuleList(torch.nn.Module):
    # [CLASS DOC] Module wrapping a ModuleList and defining its own getitem and length so that
    # indexing and len go through user-defined methods.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a ModuleList of four layers (Linear, ReLU, Linear,
        # ReLU).
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    def __getitem__(self, idx: int):
        # [FUNCTION DOC] HELPER: Returns the layer at the given integer index of the wrapped
        # list.
        return self.layers[idx]

    def __len__(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the number of wrapped layers.
        return len(self.layers)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Loops over indices up to the module length, applying each layer
        # obtained via the module's own getitem; returns the result.
        for i in range(len(self)):
            x = self[i](x)

        return x


class ModuleDict(torch.nn.Module):
    # [CLASS DOC] Module holding a ModuleDict with one Linear layer under the key "0".
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a ModuleDict containing a single Linear layer under
        # "0".
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Looks up the layer under the string key "0" and applies it to
        # x. A TODO notes that more logic is to be handled.
        # TODO(future PR): handle more logic
        x = self.layers["0"](x)
        return x


class ParameterDict(torch.nn.Module):
    # [CLASS DOC] Module holding a ParameterDict with one random matrix parameter under the key
    # "0".
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a ParameterDict with one 10 by 10 Parameter under "0".
        super().__init__()
        self.layers = torch.nn.ParameterDict(
            {
                "0": torch.nn.Parameter(torch.randn(10, 10)),
            }
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Matrix-multiplies the stored parameter by x and returns the
        # result.
        x = self.layers["0"].mm(x)
        return x


class CustomGetItemParameterDict(torch.nn.Module):
    # [CLASS DOC] Module wrapping a ParameterDict and defining its own getitem that delegates to
    # it.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a ParameterDict with one 10 by 10 Parameter under "0".
        super().__init__()
        self.layers = torch.nn.ParameterDict(
            {
                "0": torch.nn.Parameter(torch.randn(10, 10)),
            }
        )

    def __getitem__(self, key: str) -> torch.nn.Module:
        # [FUNCTION DOC] HELPER: Returns the entry stored under the key in the wrapped dict.
        # KNOWN ISSUE: the return annotation says Module although the stored value is a
        # Parameter.
        return self.layers[key]

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Fetches the parameter through the module's own getitem with key
        # "0" and matrix-multiplies it by x.
        x = self["0"].mm(x)
        return x


class CustomGetItemModuleDict(torch.nn.Module):
    # [CLASS DOC] Module wrapping a ModuleDict and defining its own getitem that delegates to
    # it.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a ModuleDict with one Linear layer under "0".
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )

    def __getitem__(self, key: str) -> torch.nn.Module:
        # [FUNCTION DOC] HELPER: Returns the submodule stored under the key in the wrapped dict.
        return self.layers[key]

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Fetches the layer through the module's own getitem with key "0"
        # and applies it to x.
        x = self["0"](x)
        return x


class TensorList(torch.nn.Module):
    # [CLASS DOC] Module storing a tuple of four tensors with alternating shapes (one by ten and
    # ten by one) and multiplying the input by each.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Stores a tuple of four random tensors as attribute layers.
        super().__init__()
        self.layers = (
            torch.randn((1, 10)),
            torch.randn((10, 1)),
            torch.randn((1, 10)),
            torch.randn((10, 1)),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Multiplies x elementwise by each stored tensor in turn;
        # broadcasting yields a ten by ten result.
        for layer in self.layers:
            x = x * layer
        return x


class Children(torch.nn.Module):
    # [CLASS DOC] Module iterating over its registered children in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Registers two Linear and two ReLU children l1 to l4 in
        # alternating order.
        super().__init__()
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies each child from children to x in registration order and
        # returns the result.
        for block in self.children():
            x = block(x)
        return x


class NamedChildren(torch.nn.Module):
    # [CLASS DOC] Module iterating over named_children, ignoring names, in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Registers two Linear and two ReLU children l1 to l4 in
        # alternating order.
        super().__init__()
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies each child yielded by named_children to x in order and
        # returns the result.
        for _, block in self.named_children():
            x = block(x)
        return x


class IntArg(torch.nn.Module):
    # [CLASS DOC] Module whose forward takes an extra integer argument with a default value.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1.
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)

    def forward(self, x, offset=1):
        # [FUNCTION DOC] HELPER: Returns relu of layer1 applied to x, plus the offset (default
        # one).
        x = F.relu(self.layer1(x)) + offset
        return x


class Seq(torch.nn.Module):
    # [CLASS DOC] Module wrapping a Sequential of Linear, ReLU, Linear, ReLU.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layers as a Sequential of four modules.
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the Sequential applied to x.
        return self.layers(x)


class Cfg:
    # [CLASS DOC] Plain (non-Module) config object with two attributes, val and count.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets val to 0.5 and count to 3.
        self.val = 0.5
        self.count = 3


class CfgModule(torch.nn.Module):
    # [CLASS DOC] Module that reads constants from a plain config object stored as an attribute.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a Cfg instance and a Linear layer.
        super().__init__()
        self.cfg = Cfg()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Loops cfg.count times, each time adding cfg.val to x and
        # applying the layer; returns the result.
        for _ in range(self.cfg.count):
            x = self.layer(x + self.cfg.val)
        return x


class StringMember(torch.nn.Module):
    # [CLASS DOC] Module with a string attribute that forward compares against a literal.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1 and sets mode to a fixed string.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.mode = "some_string"

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns relu of linear1 applied to x when mode equals the
        # expected string; otherwise returns None implicitly.
        if self.mode == "some_string":
            return F.relu(self.linear1(x))


class _Block(torch.nn.Module):
    # [CLASS DOC] HELPER: Tiny module used inside dense-block containers; mimics a layer that
    # consumes a list of tensors.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Concatenates the input list of tensors along dimension one and
        # scales by 1.5.
        return 1.5 * torch.cat(x, 1)


class _DenseBlock(torch.nn.ModuleDict):
    # [CLASS DOC] HELPER: ModuleDict subclass that mimics a DenseNet block; registers num_layers
    # _Block modules and concatenates their outputs.
    _version = 2

    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        # [FUNCTION DOC] HELPER: Registers num_layers _Block modules named denselayer1 and
        # onward, using the default of three layers.
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        # [FUNCTION DOC] HELPER: Keeps a growing list of feature tensors starting with the
        # input; each layer (iterated via values) consumes the list and its output is appended;
        # returns all features concatenated along dimension one.
        features = [init_features]
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNetBlocks(torch.nn.Module):
    # [CLASS DOC] Module wrapping a _DenseBlock to test iteration over ModuleDict values.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a _DenseBlock with default layer count.
        super().__init__()
        self.layers = _DenseBlock()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the dense block applied to x.
        return self.layers(x)


class MaterializedModule(torch.nn.Module):
    # [CLASS DOC] Materialized form of the lazy module below: a module with a single optional
    # parameter that initially is None, and an identity forward.
    """Once the below lazy module is initialized with its first input,
    it is transformed into this module."""

    param: Parameter

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Registers a parameter named param with value None.
        super().__init__()
        self.register_parameter("param", None)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns x unchanged.
        return x


class LazyModule(LazyModuleMixin, MaterializedModule):
    # [CLASS DOC] Lazy module that becomes a MaterializedModule on first call, once
    # initialize_parameters has materialized its uninitialized parameter.
    param: UninitializedParameter
    cls_to_become = MaterializedModule

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the parent constructor and sets param to an
        # UninitializedParameter.
        super().__init__()
        self.param = UninitializedParameter()

    def initialize_parameters(self, x):
        # [FUNCTION DOC] HELPER: Forces a Dynamo graph break (so the call cannot be inlined) and
        # then materializes param with the shape of the input tensor.
        # force graph break to ensure this was not inlined
        torch._dynamo.graph_break()
        self.param.materialize(x.shape)


class LazyMLP(torch.nn.Module):
    # [CLASS DOC] Module with two LazyLinear layers (output sizes 10 and 1) separated by ReLUs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates fc1 and fc2 as LazyLinear modules with ReLU activations
        # relu1 and relu2.
        super().__init__()
        self.fc1 = torch.nn.LazyLinear(10)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.LazyLinear(1)
        self.relu2 = torch.nn.ReLU()

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Applies fc1, relu1, fc2, relu2 in order and returns the final
        # tensor.
        x = self.relu1(self.fc1(input))
        y = self.relu2(self.fc2(x))
        return y


class MyInput(NamedTuple):
    # [CLASS DOC] Named tuple with a nested mapping field x (strings to mappings of tensors) and
    # a tensor field y.
    x: dict[str, dict[str, torch.Tensor]]
    y: torch.Tensor


class LazyLayerWithNamedTupleInput(LazyModuleMixin, torch.nn.Module):
    # [CLASS DOC] Lazy layer whose input is a named tuple; its parameter is shaped from the
    # first element of the list stored under key a of field x.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs only the parent constructor.
        super().__init__()

    def initialize_parameters(self, input):
        # [FUNCTION DOC] HELPER: Under no_grad, creates parameter _param as a tensor of the
        # shape of the first element of input.x under key a, filled with 0.5.
        with torch.no_grad():
            self._param = torch.nn.Parameter(
                torch.empty(input.x["a"][0].shape).fill_(0.5)
            )

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Sums all tensors in the list stored at input.x under key a and
        # returns the sum.
        input = input.x["a"]
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithNamedTupleInput(torch.nn.Module):
    # [CLASS DOC] Wrapper that holds a LazyLayerWithNamedTupleInput and forwards the named tuple
    # input to it.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the lazy layer as attribute layer.
        super().__init__()
        self.layer = LazyLayerWithNamedTupleInput()

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Returns the layer called on the input.
        return self.layer(input)


class LazyLayerWithListInput(LazyModuleMixin, torch.nn.Module):
    # [CLASS DOC] Lazy layer whose input is a list of tensors; the parameter is shaped like the
    # first element.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs only the parent constructor.
        super().__init__()

    def initialize_parameters(self, input):
        # [FUNCTION DOC] HELPER: Under no_grad, creates parameter _param shaped like input
        # element zero, filled with 0.5.
        with torch.no_grad():
            self._param = torch.nn.Parameter(torch.empty(input[0].shape).fill_(0.5))

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Sums every element of the input list and returns the sum.
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithListInput(torch.nn.Module):
    # [CLASS DOC] Wrapper that drops the last element of its input list before passing it to a
    # LazyLayerWithListInput.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the lazy layer as attribute layer.
        super().__init__()
        self.layer = LazyLayerWithListInput()

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Calls the layer on all but the last element of the input (the
        # tests append None as a trailing element).
        return self.layer(input[:-1])


class LazyModuleWithLazySubmodule(LazyModuleMixin, torch.nn.Module):
    # [CLASS DOC] Lazy module that creates a lazy submodule while initializing its parameters.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs only the parent constructor.
        super().__init__()

    def initialize_parameters(self, input):
        # [FUNCTION DOC] HELPER: Under no_grad, assigns a new LazyLayerWithListInput as
        # attribute layer.
        with torch.no_grad():
            self.layer = LazyLayerWithListInput()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the layer called on x.
        return self.layer(x)


class LazyLayerWithInputs(LazyModuleMixin, torch.nn.Module):
    # [CLASS DOC] Lazy layer taking two list inputs; initialization creates one parameter per
    # input.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs only the parent constructor.
        super().__init__()

    def initialize_parameters(self, x, y):
        # [FUNCTION DOC] HELPER: Under no_grad, creates _param_x and _param_y shaped like the
        # first element of x and of y, both filled with 0.5.
        with torch.no_grad():
            self._param_x = torch.nn.Parameter(torch.empty(x[0].shape).fill_(0.5))
            self._param_y = torch.nn.Parameter(torch.empty(y[0].shape).fill_(0.5))

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Sums the elements of x and of y separately and returns the sum
        # of both totals.
        res_x = 0
        for i in range(len(x)):
            res_x = res_x + x[i]
        res_y = 0
        for i in range(len(y)):
            res_y = res_y + y[i]
        return res_x + res_y


class LazyModuleKwArgs(LazyModuleMixin, torch.nn.Module):
    # [CLASS DOC] Lazy module that forwards a positional and a keyword argument to a lazily
    # created submodule.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs only the parent constructor.
        super().__init__()

    def initialize_parameters(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Accepts any arguments and, under no_grad, assigns a new
        # LazyLayerWithInputs as attribute layer.
        with torch.no_grad():
            self.layer = LazyLayerWithInputs()

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Calls the layer with x positionally and y as keyword.
        return self.layer(x, y=y)


class LazyModuleBadInferParams(LazyModuleMixin, torch.nn.Module):
    # [CLASS DOC] Lazy module whose initialize_parameters is broken: it increments an attribute
    # foo that does not exist, to test error propagation.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs only the parent constructor.
        super().__init__()

    def initialize_parameters(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Increments self.foo; raises AttributeError because foo is never
        # defined.
        self.foo += 1

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Calls self.layer with x and y; never reached in tests because
        # initialization fails first.
        # KNOWN ISSUE: attribute layer is never defined on this class.
        return self.layer(x, y=y)


class LazyParentModule(LazyModuleMixin, torch.nn.Module):
    # [CLASS DOC] Lazy parent module defining a helper that depends on a value set by a
    # subclass.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs only the parent constructor.
        super().__init__()

    def impl(self, x):
        # [FUNCTION DOC] HELPER: Returns the cosine of x plus the _val attribute set by subclass
        # initialization.
        return x.cos() + self._val


class LazyChildModuleNoClsToBecome(LazyParentModule):
    # [CLASS DOC] Lazy subclass without a cls_to_become that calls the parent helper via super.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs only the parent constructor.
        super().__init__()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls the parent impl on the sine of x via super.
        return super().impl(x.sin())

    def initialize_parameters(self, input):
        # [FUNCTION DOC] HELPER: Sets _val to a Parameter of ones shaped two by two; the input
        # is ignored.
        self._val = torch.nn.Parameter(torch.ones(2, 2))


def requires_grad1(module: torch.nn.Module, recurse: bool = False) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if any parameter of the module (optionally recursive,
    # default non-recursive) requires grad.
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def requires_grad2(module: torch.nn.Module, recurse: bool = False) -> bool:
    # [FUNCTION DOC] HELPER: Duplicate of requires_grad1 under another name; returns True if any
    # parameter requires grad, with optional recursion defaulting to off.
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


class ParametersModule1(torch.nn.Module):
    # [CLASS DOC] Module with a Linear layer and a Parameter scale; forward branches on whether
    # any direct parameter requires grad through requires_grad1.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1 and a random one-by-ten Parameter scale.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: When no direct parameter requires grad, returns relu of linear1
        # times scale; otherwise returns x plus one. Because Parameters require grad by default,
        # the second branch is normally taken.
        if not requires_grad1(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule2(ParametersModule1):
    # [CLASS DOC] Variant of ParametersModule1 whose forward uses requires_grad2 instead.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Same branching as the parent but using requires_grad2.
        if not requires_grad2(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule3(ParametersModule1):
    # [CLASS DOC] Variant that builds a ones tensor with the dtype of the first parameter from
    # parameters.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Builds a ones tensor of length ten with the dtype of the first
        # parameter found in parameters, then returns relu of linear1 times scale plus ones.
        ones = torch.ones(10, dtype=next(self.parameters()).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule4(ParametersModule1):
    # [CLASS DOC] Variant of ParametersModule1 that reads the first direct parameter via
    # parameters with recurse turned off.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Builds a ones tensor of length ten with the dtype of the first
        # non-recursive parameter, then returns relu of linear1 times scale plus ones.
        ones = torch.ones(10, dtype=next(self.parameters(recurse=False)).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule5(torch.nn.Module):
    # [CLASS DOC] Module whose scale parameter is registered under two attribute names (scale
    # and scale_dup), to test that parameters deduplicates.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1 and a ten by ten Parameter scale, and aliases
        # it as scale_dup.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.nn.Parameter(torch.randn(10, 10))
        self.scale_dup = self.scale

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Counts the parameters yielded by parameters (shared parameter
        # counted once) and returns x times scale times that count.
        counter = 0
        for _param in self.parameters():
            counter += 1

        return x * self.scale * counter


class SuperModule(BasicModule):
    # [CLASS DOC] BasicModule subclass whose forward calls the parent forward via super and adds
    # ten.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the parent forward result plus 10.0.
        x = super().forward(x)
        return x + 10.0


class SuperModule2(BasicModule):
    # [CLASS DOC] BasicModule subclass whose forward calls BasicModule.forward explicitly with
    # self instead of super.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the parent forward result obtained through an explicit
        # unbound call.
        return BasicModule.forward(self, x)


class ComplicatedSuperParent(torch.nn.Module):
    # [CLASS DOC] Parent module with a class method that doubles its input.
    @classmethod
    def custom_add(cls, x):
        # [FUNCTION DOC] HELPER: Class method returning x plus x.
        x = x + x
        return x


class SuperChildCallsClassMethod(ComplicatedSuperParent):
    # [CLASS DOC] Subclass whose class method calls the parent's class method through super.
    @classmethod
    def child_func(cls, x):
        # [FUNCTION DOC] HELPER: Class method returning custom_add (resolved via super) applied
        # to x.
        x = super().custom_add(x)
        return x

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies child_func to x and returns the result.
        x = self.child_func(x)
        return x


class HasAttrModule(torch.nn.Module):
    # [CLASS DOC] Module with a Parameter scale; forward uses hasattr to conditionally apply
    # scale and a nonexistent scale2.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates scale as a one-by-ten Parameter.
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies relu, then multiplies in place by scale if the
        # attribute exists; the scale2 branch is skipped because that attribute does not exist.
        x = F.relu(x)
        if hasattr(self, "scale"):
            x *= self.scale
        if hasattr(self, "scale2"):
            x *= self.scale2
        return x


class EnumValues(torch.nn.ModuleDict):
    # [CLASS DOC] ModuleDict subclass registering num_layers _Block modules, whose forward
    # iterates values (dense-block style).
    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        # [FUNCTION DOC] HELPER: Registers num_layers _Block modules named denselayer1 onward
        # (default three).
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        # [FUNCTION DOC] HELPER: Appends each layer's output (computed from the running feature
        # list) and returns all features concatenated along dimension one.
        features = [init_features]
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class AccessByKeys(torch.nn.ModuleDict):
    # [CLASS DOC] ModuleDict subclass like EnumValues but iterating keys and indexing itself.
    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        # [FUNCTION DOC] HELPER: Registers num_layers _Block modules named denselayer1 onward
        # (default three).
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        # [FUNCTION DOC] HELPER: For each key, indexes the module to get the layer, appends its
        # output on the feature list, and returns all features concatenated along dimension one.
        features = [init_features]
        for k in self.keys():
            new_features = self[k](features)
            features.append(new_features)
        return torch.cat(features, 1)


class CallForwardDirectly(torch.nn.Module):
    # [CLASS DOC] Module that invokes submodule forward methods directly instead of calling the
    # submodules.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer1 as BasicModule and layer2 as Linear.
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls forward directly on layer1 then layer2 and returns the
        # result.
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return x


class ConvCallForwardDirectly(torch.nn.Module):
    # [CLASS DOC] Module that calls forward directly on a Conv2d layer.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a bias-free Conv2d layer with three input channels, 64
        # output channels, kernel three, stride one, padding one.
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the conv layer's forward applied to x.
        return self.layer.forward(x)


class ConvTransposeCallForwardDirectly(torch.nn.Module):
    # [CLASS DOC] Module that calls forward directly on a ConvTranspose2d layer.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a ConvTranspose2d layer with four channels in and out
        # and kernel four.
        super().__init__()
        self.layer = torch.nn.ConvTranspose2d(4, 4, 4)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the layer's forward applied to x.
        return self.layer.forward(x)


class ConvCallSuperForwardDirectly(torch.nn.Conv1d):
    # [CLASS DOC] Conv1d subclass whose forward calls the parent forward through super.
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards the channel counts, kernel size, and any extra keyword
        # arguments to the Conv1d constructor.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, inputs, mask=None):
        # [FUNCTION DOC] HELPER: Returns the parent forward on inputs; the mask argument is
        # accepted but ignored.
        outputs = super().forward(inputs)
        return outputs


class ConvTransposeCallSuperForwardDirectly(torch.nn.ConvTranspose2d):
    # [CLASS DOC] ConvTranspose2d subclass with a forward that handles empty input specially (as
    # seen in detection code).
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards the channel counts, kernel size, and extra keyword
        # arguments to the ConvTranspose2d constructor.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: If x has elements, returns the parent forward. Otherwise
        # computes an expected output spatial shape from stride, padding, dilation, kernel and
        # output padding and calls a helper to create an empty tensor.
        # KNOWN ISSUE: the name _NewEmptyTensorOp is not defined anywhere (suppressed with
        # noqa), so the empty-input path would raise NameError; the tests only use non-empty
        # input.
        if x.numel() > 0:
            return super().forward(x)
        output_shape = [
            ((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op)
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)  # noqa: F821


class ModuleNameString(torch.nn.Module):
    # [CLASS DOC] Module that compares class names as strings in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear1.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns 10 if its own class name is ABC (never true here);
        # returns relu of linear1 of x plus ten when the linear layer's class name is Linear;
        # else returns 11.
        if self.__class__.__name__ == "ABC":
            return 10
        if self.linear1.__class__.__name__ == "Linear":
            return F.relu(self.linear1(x) + 10)
        return 11


class SelfMutatingModule(torch.nn.Module):
    # [CLASS DOC] Module wrapping a given layer and mutating an integer counter on every forward
    # call.
    def __init__(self, layer):
        # [FUNCTION DOC] HELPER: Stores the given layer and sets counter to zero.
        super().__init__()
        self.layer = layer
        self.counter = 0

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds the current counter to the layer output, increments
        # counter, and returns relu of the result.
        result = self.layer(x) + self.counter
        self.counter += 1
        return F.relu(result)


class ModuleAttributePrecedenceBase(torch.nn.Module):
    # [CLASS DOC] Base module defining a method named linear that is later shadowed by an
    # instance attribute in a subclass.
    def linear(self, x, flag=None):
        # [FUNCTION DOC] HELPER: Returns x times 2.0 when flag is truthy, otherwise x times 3.0.
        if flag:
            return x * 2.0
        return x * 3.0


class ModuleAttributePrecedence(ModuleAttributePrecedenceBase):
    # [CLASS DOC] Module testing attribute precedence: instance attributes named activation,
    # linear, initializer, and scale shadow methods with the same names defined on the class or
    # its base.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets activation (ReLU module), linear (Linear module),
        # initializer (ones tensor) and scale (0.5) as instance attributes, shadowing the same-
        # named methods.
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(10, 10)
        self.initializer = torch.ones([10, 10])
        self.scale = 0.5

    def activation(self, x):
        # [FUNCTION DOC] HELPER: Method shadowed by the activation instance attribute; never
        # called. Would return x times 1.2.
        return x * 1.2

    def initializer(self):
        # [FUNCTION DOC] HELPER: Method shadowed by the initializer instance attribute; never
        # called. Would return a zeros tensor.
        return torch.zeros([10, 10])

    def scale(self):
        # [FUNCTION DOC] HELPER: Method shadowed by the scale instance attribute; never called.
        # Would return 2.0.
        return 2.0

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds the initializer tensor to x, applies linear then
        # activation, and multiplies by scale, using the instance attributes.
        # object attribute takes precedence unless it's a nn.Module
        return self.activation(self.linear(self.initializer + x)) * self.scale


class ModuleForwardHasGraphBreak(torch.nn.Module):
    # [CLASS DOC] Module whose forward collects parameters, buffers and modules (as dicts and
    # lists), forces a graph break, then uses the collected values afterwards, to test
    # reconstruction after a break.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds layer1, layer2, a Sequential layer3 of two BasicModules,
        # a ModuleList layer4, a ModuleDict layer5 and a scale tensor.
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.layer3 = torch.nn.Sequential(BasicModule(), BasicModule())
        self.layer4 = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )
        self.layer5 = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs layer1, collects named and unnamed parameters, buffers and
        # modules, graph-breaks, then reassigns y to the collected parameter mapping and sums
        # layer2 output with three weights looked up by name; returns that times scale. The
        # docstring explains this checks reconstruction of those containers after a graph break
        # (linked issue 1931).
        """
        This is used to test if the results of functions like `named_parameters`
        can be reconstructed correctly after graph break.

        https://github.com/pytorch/torchdynamo/issues/1931
        """
        x = self.layer1(x)
        params1 = dict(self.named_parameters())
        params2 = list(self.parameters())
        buffers1 = dict(self.named_buffers())
        buffers2 = list(self.buffers())
        modules1 = dict(self.named_modules())
        modules2 = list(self.modules())
        torch._dynamo.graph_break()
        y = modules2
        y = modules1
        y = buffers2
        y = buffers1
        y = params2
        y = params1
        x = (
            self.layer2(x)
            + y["layer3.1.linear1.weight"]
            + y["layer4.2.weight"]
            + y["layer5.0.weight"]
        )
        return x * self.scale


class ModuleGuardNameIsValid(torch.nn.ModuleDict):
    # [CLASS DOC] ModuleDict subclass whose module names contain special characters, to check
    # that guard names for such paths are valid identifiers.
    # Guard names should be valid python identifier as we use eval() to get
    # corresponding guard value. Some guard names come from source(module path)
    # where special symbols are valid. But they are not valid python identifier,
    # we should identify these pattern and rewrite them with getattr.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Registers two BasicModule children named with at-sign and
        # hyphen characters.
        super().__init__()
        for i in range(2):
            self.add_module(f"l@yer-{i + 1:d}", BasicModule())

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies each child from values in order and returns the result.
        for layer in self.values():
            x = layer(x)
        return x


class SequentialWithDuplicatedModule(torch.nn.Module):
    # [CLASS DOC] Module holding a Sequential in which the same ReLU module appears three times.
    # Sequential module(self.layer) contains three duplicated ReLU module.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates relu and a Sequential of Linear, relu, Linear, relu,
        # Linear, relu using the same relu object each time.
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            self.relu,
            torch.nn.Linear(20, 20),
            self.relu,
            torch.nn.Linear(20, 10),
            self.relu,
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the Sequential applied to x.
        return self.layer(x)


class SequentialWithDuplicatedModule2(torch.nn.Module):
    # [CLASS DOC] Like SequentialWithDuplicatedModule but the Sequential is built from an
    # ordered mapping with named entries.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates relu and a Sequential from an OrderedDict of named
        # linear and relu entries, reusing the same relu object.
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("linear1", torch.nn.Linear(10, 20)),
                    ("relu1", self.relu),
                    ("linear2", torch.nn.Linear(20, 20)),
                    ("relu2", self.relu),
                    ("linear3", torch.nn.Linear(20, 10)),
                    ("relu3", self.relu),
                ]
            )
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the Sequential applied to x.
        return self.layer(x)


class ModuleComparison(torch.nn.Module):
    # [CLASS DOC] Module with three Linear layers exposed as a list by a property and compared
    # in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer0, layer1, layer2 as Linear layers of size ten.
        super().__init__()
        self.layer0 = torch.nn.Linear(10, 10)
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

    @property
    def encoder_layers(self):
        # [FUNCTION DOC] HELPER: Property returning a new list of the three layers.
        return [self.layer0, self.layer1, self.layer2]

    def forward(self, x):
        # [FUNCTION DOC] HELPER: For each layer, applies it to x and uses relu6 if the layer is
        # None or equals layer0, else relu; returns the output of the last iteration.
        for layer in self.encoder_layers:
            output = layer(x)
            if layer is None or layer == self.layer0:
                output = F.relu6(output)
            else:
                output = F.relu(output)
        return output


class ModulePatch1(torch.nn.Module):
    # [CLASS DOC] Empty Module subclass without a forward; tests patch a forward onto an
    # instance.
    pass


class ModulePatch2(torch.nn.Module):
    # [CLASS DOC] Module with forward returning x minus one; used as a source of a method to
    # patch onto ModulePatch1 instances.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns x minus 1.
        return x - 1


class UnspecNonInlinableModule(torch.nn.Module):
    # [CLASS DOC] Module marked with torchdynamo_force_dynamic so that Dynamo treats it as an
    # unspecialized nn module; forward branches on the sum of x.
    torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns x plus one when the sum of x is positive, otherwise x
        # minus one.
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1


class UnspecNonInlinableToplevelModule(torch.nn.Module):
    # [CLASS DOC] Wrapper holding an UnspecNonInlinableModule.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the inner module as attribute m.
        super().__init__()
        self.m = UnspecNonInlinableModule()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the inner module applied to x.
        return self.m(x)


class ModuleWithIntAttr(torch.nn.Module):
    # [CLASS DOC] Module with a Linear layer and an integer attribute step added in forward.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates a Linear of size four layer and sets step to ten.
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self.step = 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Adds one to x, applies the layer, and adds step.
        x = x + 1
        return self.layer(x) + self.step


class UnspecInlinableModule(torch.nn.Module):
    # [CLASS DOC] Module marked with torchdynamo_force_dynamic (unspecialized nn module) with a
    # simple sine forward.
    torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the sine of x.
        return torch.sin(x)


class UnspecModuleWithIntAttr(torch.nn.Module):
    # [CLASS DOC] Wrapper around UnspecInlinableModule with an integer attribute step.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the inner layer and sets step to ten.
        super().__init__()
        self.layer = UnspecInlinableModule()
        self.step = 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Adds one to x, applies the inner layer, and adds step.
        x = x + 1
        return self.layer(x) + self.step


def make_test(fn, expected_ops=None):
    # [FUNCTION DOC] HELPER: Builds a test method for a module: the returned function runs the
    # Dynamo standard_test helper on that module with one argument and the given expected op
    # count. Also puts the module in eval mode immediately (side effect on fn).
    def test_fn(self):
        # [FUNCTION DOC] HELPER: Test body returned by make_test; delegates to standard_test
        # with the captured module, one argument, and expected_ops; returns its result.
        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=expected_ops
        )

    fn.eval()
    return test_fn


def temporary_tensor_subclass(torch_function=None):
    # [FUNCTION DOC] HELPER: Builds a Tensor subclass whose torch function handler optionally
    # invokes a callback before delegating to the default handler; returns the class.
    class TensorProxy(torch.Tensor):
        # [CLASS DOC] HELPER: Local Tensor subclass created per call, closing over
        # torch_function.
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            # [FUNCTION DOC] HELPER: torch function handler that calls the optional
            # torch_function callback if provided, then returns the parent handler result.
            if torch_function is not None:
                torch_function()
            return super().__torch_function__(func, types, args, kwargs)

    return TensorProxy


class NNModuleTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Main test case: attributes named test_* are generated through make_test for
    # many modules (each compiles the module via standard_test and compares to eager), followed
    # by hand-written tests of module-related Dynamo behavior.
    test_seq = make_test(Seq())
    test_basicmodule1 = make_test(BasicModule())
    test_basicmodule2 = make_test(BasicModule())
    test_submodules1 = make_test(SubmoduleExample())
    test_submodules2 = make_test(SubmoduleExample())
    test_modulemethod1 = make_test(ModuleMethodCall())
    test_modulemethod2 = make_test(ModuleMethodCall())
    test_module_call_module_with_static_forward = make_test(
        ModuleCallModuleWithStaticForward()
    )
    test_module_static_method = make_test(ModuleStaticMethodCall())
    test_fnmember = make_test(FnMember())
    test_fnmembercmp1 = make_test(FnMemberCmp(F.relu))
    test_fnmembercmp2 = make_test(FnMemberCmp(None))
    test_constloop = make_test(ConstLoop())
    test_istraining1 = make_test(IsTrainingCheck())
    test_istraining2 = make_test(IsTrainingCheck())
    test_iseval1 = make_test(IsEvalCheck())
    test_iseval2 = make_test(IsEvalCheck())
    test_viamodulecall = make_test(ViaModuleCall())
    test_isnonelayer = make_test(IsNoneLayer())
    test_layerlist = make_test(LayerList())
    test_tensorlist = make_test(TensorList())
    test_intarg = make_test(IntArg())
    test_cfgmod = make_test(CfgModule())
    test_stringmember = make_test(StringMember())
    test_modulelist = make_test(ModuleList())
    test_modulelist_nested = make_test(NestedModuleList())
    test_modulelist_custom = make_test(CustomGetItemModuleList())
    test_moduledict = make_test(ModuleDict())
    test_moduledict_custom = make_test(CustomGetItemModuleDict())
    test_parameterdict = make_test(ParameterDict())
    test_parameterdict_custom = make_test(CustomGetItemParameterDict())
    test_super1 = make_test(SuperModule())
    test_super2 = make_test(SuperModule2())
    test_super_class_method = make_test(SuperChildCallsClassMethod())
    test_children = make_test(Children())
    test_named_children = make_test(NamedChildren())
    test_densenet = make_test(DenseNetBlocks())
    test_parameters1 = make_test(ParametersModule1())
    test_parameters2 = make_test(ParametersModule2())
    test_parameters3 = make_test(ParametersModule3(), expected_ops=5)
    test_parameters4 = make_test(ParametersModule4())
    test_parameters5 = make_test(ParametersModule5())
    test_hasattr = make_test(HasAttrModule())
    test_enumvalues = make_test(EnumValues())
    test_access_by_keys = make_test(AccessByKeys())
    test_module_class_method = make_test(ModuleClassMethodCall())
    test_module_property = make_test(ModuleProperty())
    test_forward_directly = make_test(CallForwardDirectly())
    test_module_name_string = make_test(ModuleNameString())
    test_module_attribute_precedence = make_test(ModuleAttributePrecedence())
    test_module_guard_name_is_valid = make_test(ModuleGuardNameIsValid())
    test_sequential_with_duplicated_module = make_test(SequentialWithDuplicatedModule())
    test_sequential_with_duplicated_module2 = make_test(
        SequentialWithDuplicatedModule2()
    )
    test_module_comparison = make_test(ModuleComparison())

    def test_inject_module_parameters(self):
        # [FUNCTION DOC] HELPER: Test that compiling with the inductor backend works when module
        # _parameters dicts are replaced by an OrderedDict subclass; it injects the replacement
        # into every submodule of ParametersModule5, compiles, and runs once, expecting no
        # error.
        from collections import OrderedDict

        class ZeROOrderedDict(OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass that remembers a parent module and
            # returns items on getitem.
            def __init__(self, parent_module=None, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the OrderedDict constructor with the remaining
                # arguments and records parent_module.
                super().__init__(*args, **kwargs)
                self._parent_module = parent_module

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Returns the item stored under key, via the parent
                # implementation.
                param = super().__getitem__(key)
                return param

        def inject_parameters(module, cls):
            # [FUNCTION DOC] HELPER: Replaces each submodule's parameter dict with a new
            # instance of cls, copying existing entries; ZeROOrderedDict receives the module as
            # parent.
            for m in module.modules():
                if cls == ZeROOrderedDict:
                    new_param = cls(parent_module=m)
                else:
                    new_param = cls()

                for key, param in m._parameters.items():
                    new_param[key] = param
                m._parameters = new_param

        model = ParametersModule5()
        inject_parameters(model, ZeROOrderedDict)
        model = torch.compile(model, backend="inductor")
        x = torch.ones(10)
        # model can be compiled without error
        y = model(x)

    def test_module_forward_has_graph_break(self):
        # [FUNCTION DOC] Test that a module whose forward gets graph breaks while holding
        # collected parameter, buffer, and module containers compiles with the eager backend and
        # matches eager output.
        m = ModuleForwardHasGraphBreak()
        x = torch.rand([10, 10])
        ref = m(x)
        opt_m = torch.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_unsupportedmethod(self):
        # [FUNCTION DOC] Test that a module with an unsupported call inside a method compiles
        # with a counting backend, matches eager output, and produces five operations in total.
        m = UnsupportedMethodCall()
        i = torch.randn(10)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt)
        r = opt_m(i)
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 5)

    def test_unsupportedmodule(self):
        # [FUNCTION DOC] Test that a wrapper around a module with an unsupported call compiles,
        # matches eager output, and produces six operations in total.
        m = UnsupportedModuleCall()
        i = torch.randn(10)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt)
        r = opt_m(i)
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 6)

    @patch.object(torch._dynamo.config, "allow_unspec_int_on_nn_module", True)
    def test_self_mutating1(self):
        # [FUNCTION DOC] Test that a module that increments an integer counter in forward
        # compiles consistently across calls with optimize_assert, matching three eager calls;
        # the expected frame count is one with dynamic shapes and two with static default.
        # Allows unspecialized integers on nn modules via a config patch.
        m1 = torch.nn.Linear(10, 10)
        m2 = SelfMutatingModule(m1)
        m3 = SelfMutatingModule(m1)
        m4 = SelfMutatingModule(m1)
        i = torch.randn(10)
        out2 = [m2(i), m2(i), m2(i)]
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m3 = torch._dynamo.optimize_assert(cnt)(m3)
        opt_m4 = torch._dynamo.optimize_assert(cnt)(m4)
        out3 = [opt_m3(i), opt_m3(i), opt_m3(i)]
        out4 = [opt_m4(i), opt_m4(i), opt_m4(i)]
        self.assertTrue(torch._dynamo.testing.same(out2, out3))
        self.assertTrue(torch._dynamo.testing.same(out2, out4))
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_nn_module_setattr(self):
        # [FUNCTION DOC] Test that mutating a plain attribute on a module after compilation
        # triggers recompilation and the new value is used.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer attribute var initially zero.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets var to zero.
                super().__init__()
                self.var = 0

        @torch.compile(backend="eager", dynamic=False)
        def f(x, m):
            # [FUNCTION DOC] HELPER: Compiled function (eager backend, static) returning x plus
            # the module's var.
            return x + m.var

        inp = torch.ones(3)
        m = Mod()

        self.assertEqual(f(inp, m), inp)
        # In 3.13.0, setattr will not fire a __dict__'s watchers,
        # so guards may not be invalidated.
        m.var = 1
        # should trigger a recompile
        self.assertEqual(f(inp, m), inp + 1)

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_generation_tag(self):
        # [FUNCTION DOC] Test that GenerationTracker assigns a generation to each module:
        # instances made before and inside an optimize_assert context get consecutive values,
        # and a deepcopy of an old module gets the current generation. Patches config to not
        # raise on context-manager usage.
        cnt = torch._dynamo.testing.CompileCounter()

        # guarantee that we have installed
        # the generation tagging function
        with torch._dynamo.optimize_assert(cnt):
            pass

        m1 = torch.nn.Linear(10, 10)
        prev_generation = GenerationTracker.get_generation_value(m1)
        cur_generation = prev_generation + 1

        with torch._dynamo.optimize_assert(cnt):
            m2 = torch.nn.Linear(10, 10)

        self.assertEqual(GenerationTracker.get_generation_value(m1), prev_generation)
        self.assertEqual(GenerationTracker.get_generation_value(m2), cur_generation)
        # check that newly constructed instances
        # also have the same generation (even if copied from an old instance)
        m3 = deepcopy(m1)
        self.assertEqual(GenerationTracker.get_generation_value(m3), cur_generation)

    def test_simple_torch_function(self):
        # [FUNCTION DOC] Test a tensor subclass with a torch function handler is traced with
        # fullgraph, with four operations, matching eager.
        def foo(x):
            # [FUNCTION DOC] HELPER: Applies sigmoid as function twice and as method twice.
            # function call, twice to test wrapping
            x = F.sigmoid(x)
            x = F.sigmoid(x)
            # method call, twice to test wrapping
            x = x.sigmoid()
            x = x.sigmoid()
            return x

        TensorProxy = temporary_tensor_subclass()
        x = torch.randn(1).as_subclass(TensorProxy)
        cnt = torch._dynamo.testing.CompileCounter()
        out1 = foo(x)
        opt_foo = torch.compile(foo, backend=cnt, fullgraph=True)
        out2 = opt_foo(x)

        self.assertEqual(cnt.op_count, 4)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

    def test_torch_function_with_closure(self):
        # [FUNCTION DOC] Test that a torch function callback that reads a closure variable is
        # traced correctly in fullgraph mode with four operations.
        def run():
            # [FUNCTION DOC] HELPER: Outer helper which defines and runs the whole scenario.
            def foo(x):
                # [FUNCTION DOC] HELPER: Applies sigmoid twice as function and twice as method.
                # function call, twice to test wrapping
                x = F.sigmoid(x)
                x = F.sigmoid(x)
                # method call, twice to test wrapping
                x = x.sigmoid()
                x = x.sigmoid()
                return x

            counter = 0

            def function():
                # [FUNCTION DOC] HELPER: Callback reading the nonlocal counter (reads only,
                # writes unsupported per the TODO).
                nonlocal counter
                # for now, only support reads from closure cells
                # TODO(future PR): support writes as well
                counter + 1

            TensorProxy = temporary_tensor_subclass(function)
            x = torch.randn(1).as_subclass(TensorProxy)
            x = torch.randn(1)
            cnt = torch._dynamo.testing.CompileCounter()
            out1 = foo(x)
            opt_foo = torch.compile(foo, backend=cnt, fullgraph=True)
            out2 = opt_foo(x)

            self.assertEqual(cnt.op_count, 4)
            self.assertTrue(torch._dynamo.testing.same(out1, out2))

        run()

    def test_torch_mangled_class_name(self):
        # [FUNCTION DOC] Test that mangled class names of tensor subclasses, collected by
        # wrapping the TensorWithTFOverrideVariable global_mangled_class_name method, have the
        # form prefix, numeric type id, underscore, compile id starting with c; three distinct
        # compile ids are expected over a graph-break function. Restores the original method
        # afterwards.
        original = TensorWithTFOverrideVariable.global_mangled_class_name
        results = []

        def instrumented(self, tx):
            # [FUNCTION DOC] HELPER: Wrapper that calls the original name-mangling method,
            # records its result in the results list, and returns it.
            result = original(self, tx)
            results.append(result)
            return result

        TensorWithTFOverrideVariable.global_mangled_class_name = instrumented

        def one_break(x):
            # [FUNCTION DOC] HELPER: Applies sigmoid, prints (forcing a graph break), then
            # applies sigmoid as method.
            x = F.sigmoid(x)
            print()  # force break
            x = x.sigmoid()
            return x

        try:
            TensorProxy = temporary_tensor_subclass()
            x = torch.randn(1).as_subclass(TensorProxy)
            x1 = one_break(x)

            cnt = torch._dynamo.testing.CompileCounter()
            opt_one_break = torch.compile(one_break, backend=cnt)
            x2 = opt_one_break(x)

            self.assertTrue(torch._dynamo.testing.same(x1, x2))
            self.assertEqual(cnt.frame_count, 2)
            self.assertEqual(cnt.op_count, 2)

            compile_ids = set()
            for r in results:
                # A mangled classname looks like __subclass_TensorProxy_94524181138240_c0
                # where the last segment contains the compile_id.
                prefix = "__subclass_TensorProxy_"
                before, sep, after = r.partition(prefix)
                self.assertEqual(before, "")
                self.assertEqual(sep, prefix)

                class_type_id, compile_id = after.split("_")
                self.assertTrue(class_type_id.isnumeric())
                self.assertTrue(compile_id.startswith("c"))

                cid = compile_id[1:]
                self.assertTrue(cid.isnumeric())
                compile_ids.add(cid)

            self.assertEqual(len(compile_ids), 3)

        finally:
            TensorWithTFOverrideVariable.global_mangled_class_name = original

    def test_nn_moduledict_contains(self):
        # [FUNCTION DOC] Test that membership tests on a ModuleDict inside forward are handled
        # by Dynamo: with the key present there are two ops, with an absent key one op (after
        # reset).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a given ModuleDict; applies a multiply only if
            # the key foo is in it, then adds one.
            def __init__(self, module_dict):
                # [FUNCTION DOC] HELPER: Stores the given module_dict.
                super().__init__()
                self.module_dict = module_dict

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies x by one if foo is in module_dict, adds one,
                # and returns it.
                if "foo" in self.module_dict:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        module_dict = torch.nn.ModuleDict({"foo": torch.nn.Conv2d(1, 1, 1)})
        m = M(module_dict)
        data = torch.randn(1)
        out1 = m(data)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        out2 = opt_m(data)
        self.assertEqual(cnt.op_count, 2)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

        module_dict = torch.nn.ModuleDict({"bar": torch.nn.Conv2d(1, 1, 1)})
        m = M(module_dict)
        data = torch.randn(1)
        out1 = m(data)
        cnt = torch._dynamo.testing.CompileCounter()
        torch._dynamo.reset()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        out2 = opt_m(data)

        self.assertEqual(cnt.op_count, 1)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

    def test_lazy_module1(self):
        # [FUNCTION DOC] Test lazy modules: a LazyModule compiled fullgraph becomes
        # MaterializedModule with param shaped like the input; the same with an unspecialized
        # module; and a torch LazyBatchNorm3d becomes BatchNorm3d with one frame and no guard
        # failures.
        input_shape = (16, 3, 6, 7, 8)

        cnt = torch._dynamo.testing.CompileCounter()
        module = LazyModule()

        def test_static_module():
            # [FUNCTION DOC] HELPER: Calls module on a ones tensor of input_shape.
            input = torch.ones(*input_shape)
            module(input)

        # test no graph break
        opt_test_static_module = torch.compile(
            test_static_module, backend=cnt, fullgraph=True
        )
        opt_test_static_module()

        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        self.assertEqual(module.param.shape, input_shape)

        # test when mapped to UnspecializedNNModule
        module = LazyModule()

        def test_unspecialized():
            # [FUNCTION DOC] HELPER: Re-creates the LazyModule in the enclosing scope and calls
            # it on a ones tensor.
            nonlocal module
            module = LazyModule()
            input = torch.ones(*input_shape)
            module(input)

        opt_test_unspecialized = torch.compile(test_unspecialized, backend=cnt)
        opt_test_unspecialized()

        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        self.assertEqual(module.param.shape, input_shape)

        # test with a static module in torch.*
        module = torch.nn.modules.LazyBatchNorm3d(
            affine=False, track_running_stats=False
        )

        cnt = torch._dynamo.testing.CompileCounter()

        torch._dynamo.reset()

        def test_torch_static():
            # [FUNCTION DOC] HELPER: Calls module on ones of input_shape and returns the output.
            input = torch.ones(*input_shape)
            return module(input)  # fully materialized

        # test no graph break
        opt_test_torch_static = torch.compile(
            test_torch_static, backend=cnt, fullgraph=True
        )
        opt_test_torch_static()
        out = opt_test_torch_static()

        self.assertTrue(same(out, module(torch.ones(*input_shape))))

        self.assertTrue(
            isinstance(module, torch.nn.modules.batchnorm.BatchNorm3d),
            "Module should be transformed to an instance of BatchNorm3d.",
        )
        self.assertEqual(cnt.frame_count, 1, "No guards should have triggered.")

    def test_lazy_module2(self):
        # [FUNCTION DOC] Test that a LazyMLP can be compiled in fullgraph eager mode before it
        # is ever run, and matches eager afterwards.
        # Test FX graph 'call_module' works well if argument is lazy module
        m = LazyMLP()
        x = torch.rand([10, 10])
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        # We should run compile mode firstly, otherwise the module
        # would be initialized when running eager mode.
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module4(self):
        # [FUNCTION DOC] Test that after a LazyMLP is initialized by a first compiled run, a
        # second input with different feature size either runs or raises a RuntimeError whose
        # traceback mentions a reduction dimension mismatch.
        m = LazyMLP()
        x = torch.rand([10, 10])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt, fullgraph=True)
        # first iteration
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        # input shape changed and second iteration
        x = torch.rand([20, 20])
        try:
            opt_m(x)
        except RuntimeError:
            self.assertIn("must have same reduction dim", traceback.format_exc())

    def test_lazy_module5(self):
        # [FUNCTION DOC] Test that a lazy module taking a list input (with trailing None)
        # compiles in fullgraph mode and matches eager.
        # Test lazy module works well with list/tuple input
        m = LazyModuleWithListInput()
        x = [torch.rand([5, 5])] * 3 + [None]
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module6(self):
        # [FUNCTION DOC] Test a lazy module that creates a lazy submodule during initialization
        # compiles in fullgraph mode and matches eager.
        # Test new lazy submodule in lazy module's initialize_parameters
        m = LazyModuleWithLazySubmodule()
        x = [torch.rand([5, 5])] * 3
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module7(self):
        # [FUNCTION DOC] Test a lazy module taking a named tuple with nested mapping input
        # compiles in fullgraph mode and matches eager.
        # Test lazy module works well with namedtuple/dict input
        m = LazyModuleWithNamedTupleInput()
        x = MyInput(
            x={"a": [torch.rand([5, 5])] * 3, "b": torch.rand([5, 5])},
            y=torch.rand([5, 5]),
        )
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module_no_cls_to_become(self):
        # [FUNCTION DOC] Test that a lazy module without cls_to_become, whose forward uses
        # super, compiles in fullgraph mode and matches eager.
        # make sure super() works in the case where cls_to_become is None
        m = LazyChildModuleNoClsToBecome()
        x = torch.rand(2, 2)
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module_kwargs(self):
        # [FUNCTION DOC] Test a lazy module receiving positional and keyword inputs compiles in
        # fullgraph mode and matches eager.
        m = LazyModuleKwArgs()
        x = [torch.rand([5, 5])] * 3
        y = [torch.rand([5, 5])] * 2
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        exp_res = m(x, y)
        self.assertTrue(torch.allclose(exp_res, opt_m(x, y)))

    def test_lazy_module_bad_params(self):
        # [FUNCTION DOC] Test that the broken initialize_parameters raises AttributeError when
        # compiling with the eager backend.
        m = LazyModuleBadInferParams()
        x = [torch.rand([5, 5])] * 3
        y = [torch.rand([5, 5])] * 2
        # Note that this raises from within dynamo code, with no exception handling.
        with self.assertRaises(AttributeError) as cm:
            opt_m = torch.compile(backend="eager")(m)
            exp_res = opt_m(x, y)

    def test_lazy_module_bad_params_call_function(self):
        # [FUNCTION DOC] Test that an AttributeError from a broken lazy module is also raised
        # when the module is called through a method of a holder class instance.
        class holder:
            # [CLASS DOC] HELPER: Holder with a class-level lazy module x.
            x = LazyModuleBadInferParams()

            def apply(self, x, y):
                # [FUNCTION DOC] HELPER: Calls the class-level lazy module with the two
                # arguments; returns nothing.
                self.x(x, y)

        def m(x, y):
            # [FUNCTION DOC] HELPER: Creates a holder and returns the result of apply.
            h = holder()
            return h.apply(x, y)

        x = [torch.rand([5, 5])] * 3
        y = [torch.rand([5, 5])] * 2
        opt_m = torch.compile(backend="eager")(m)
        with self.assertRaises(AttributeError):
            exp_res = opt_m(x, y)

    def test_lazy_module_speculation_log_divergence(self):
        # [FUNCTION DOC] Test that retracing after a graph break with a LazyLinear already
        # initialized on the first round does not diverge in the speculation log and avoids
        # recompilation: frame count stays one over two calls.
        class ModWithOneLazyLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping one LazyLinear with eight outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates layer as LazyLinear with eight outputs.
                super().__init__()
                self.layer = torch.nn.LazyLinear(8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns layer applied to x.
                return self.layer(x)

        # This allows us to restart tracing without clearing speculation log
        def id_and_fail_inlining(x):
            # [FUNCTION DOC] HELPER: Graph-breaks (forcing a restart of tracing) and returns x.
            torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def test(mod, x):
            # [FUNCTION DOC] HELPER: Compiled function that calls the module, then the graph-
            # breaking helper, and returns the module output.
            res = mod(x)
            # Speculation log must not diverge in the 2nd round of tracing,
            # after we've initialized the `LazyLinear` into a `Linear` in the
            # 1st round.
            res2 = id_and_fail_inlining(res)
            return res

        mod = ModWithOneLazyLinear()
        x = torch.ones(10, 3)

        # Make sure we don't get recompilation across multiple runs
        actual_res = test(mod, x)
        expect_res = mod(x)
        self.assertTrue(torch.allclose(expect_res, actual_res))
        actual_res = test(mod, x)
        expect_res = mod(x)
        self.assertTrue(torch.allclose(expect_res, actual_res))
        self.assertEqual(cnt.frame_count, 1)

    def test_lazy_module_symbolic_shapes(self):
        # [FUNCTION DOC] Regression test that lazy module initialization works under dynamic
        # shapes: a LazyBatchNorm1d compiled with dynamic shapes becomes BatchNorm1d with
        # num_features three and preserves output shape.
        # Regression test: lazy module initialization must handle symbolic
        # shapes from dynamic inputs without hitting "SymIntArrayRef expected
        # to contain only concrete integers".
        m = torch.nn.LazyBatchNorm1d()

        @torch.compile(backend="eager", dynamic=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning m applied to x.
            return m(x)

        x = torch.randn(16, 3, 6)
        result = fn(x)
        self.assertEqual(result.shape, x.shape)
        self.assertIsInstance(m, torch.nn.BatchNorm1d)
        self.assertEqual(m.num_features, 3)

    def test_call_fn_with_non_const_inputs_safe(self):
        # [FUNCTION DOC] Test that exporting a module whose forward goes through a helper
        # calling a conv's private forward with weights gives a graph equal to eager.
        class ModuleSpecialFwd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Conv2d and a helper that calls the conv's
            # private conv forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv as Conv2d with 3 input and 20 output
                # channels and 5 by 5 kernel.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=20, kernel_size=(5, 5)
                )

            def _conv_forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the conv's private forward with its weight and
                # bias.
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the result of the helper.
                return self._conv_forward(x)

        mod = ModuleSpecialFwd()
        rx = torch.randn([3, 10, 10])
        real = mod(rx)
        graph, _ = torch._dynamo.export(mod)(rx)
        self.assertTrue(torch._dynamo.testing.same(real, graph(rx)))

    def test_conv_call_forward_directly(self):
        # [FUNCTION DOC] Test that ConvCallForwardDirectly compiles with fullgraph and matches
        # eager.
        m = ConvCallForwardDirectly()
        x = torch.rand([4, 3, 9, 9])
        ref = m(x)
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_conv_transpose_call_forward_directly(self):
        # [FUNCTION DOC] Test that ConvTransposeCallForwardDirectly compiles with fullgraph and
        # matches eager.
        m = ConvTransposeCallForwardDirectly()
        x = torch.rand([4, 4, 4, 4])
        ref = m(x)
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_conv_call_super_forward_directly(self):
        # [FUNCTION DOC] Test that ConvCallSuperForwardDirectly compiles with fullgraph and
        # matches eager.
        x = torch.randn(4, 4)
        m = ConvCallSuperForwardDirectly(4, 4, 4)
        ref = m(x)
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_conv_transpose_call_super_forward_directly(self):
        # [FUNCTION DOC] Test that ConvTransposeCallSuperForwardDirectly compiles with fullgraph
        # and matches eager on a non-empty input.
        x = torch.randn(4, 4, 4)
        m = ConvTransposeCallSuperForwardDirectly(4, 4, 4)
        ref = m(x)
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    @torch._dynamo.config.patch("allow_unspec_int_on_nn_module", True)
    def test_nn_module_unspec_int_attr(self):
        # [FUNCTION DOC] Test over two module classes that an integer attribute is first
        # compiled as static (one frame), becomes dynamic after changing it (frame count two
        # under static default, else one), and a further change does not recompile. Allows
        # unspecialized ints via config patch.
        for module_class in [ModuleWithIntAttr, UnspecModuleWithIntAttr]:
            mod = module_class()
            cnt = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(backend=cnt)(copy.deepcopy(mod))
            x = torch.rand(3, 4)

            # Compiling `self.step` as static
            ref1 = mod(x)
            res1 = opt_mod(x)
            self.assertTrue(torch.allclose(ref1, res1))
            self.assertEqual(cnt.frame_count, 1)

            mod.step += 1
            opt_mod.step += 1

            # Second time: compiling `self.step` as dynamic
            ref2 = mod(x)
            res2 = opt_mod(x)
            self.assertTrue(torch.allclose(ref2, res2))
            self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))

            mod.step += 1
            opt_mod.step += 1

            # Third time: no re-compilation!
            ref3 = mod(x)
            res3 = opt_mod(x)
            self.assertTrue(torch.allclose(ref3, res3))
            self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))


class NNModuleTestsDevice(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-generic test class holding a lazy-module test that depends on the
    # device.
    @skipIfHpu
    def test_lazy_module3(self, device):
        # [FUNCTION DOC] Test that a LazyMLP compiled with nopython is initialized on first
        # call, then moved to the given device and rerun; outputs match eager and two frames are
        # compiled in total. Skipped on HPU.
        m = LazyMLP()
        x = torch.rand([10, 10])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        # first iteration
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        # move to device and second iteration
        m = m.to(device)
        x = x.to(device)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        self.assertEqual(cnt.frame_count, 2)


class MockModule(torch.nn.Module):
    # [CLASS DOC] Simple module with ReLU, Linear and a buffer, used in OptimizedModule tests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates relu, linear and a ten by ten buffer buf0.
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(10, 10)
        self.buf0 = torch.nn.Buffer(torch.randn(10, 10))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds the buffer to linear of x and applies relu.
        return self.relu(self.linear(x) + self.buf0)


class OptimizedModuleTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for OptimizedModule behavior, the wrapper returned when compiling an
    # nn.Module: attribute access, device and dtype moves, ordering guards, recompile limits,
    # nesting and patching.
    # KNOWN ISSUE: the class body contains a stray statement that resets the torch logging
    # settings, which executes once when the class is defined and is not part of any test.
    def test_nn_module(self):
        # [FUNCTION DOC] Test that compiling a module yields an OptimizedModule instance whose
        # output matches eager and takes one frame.
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnt)
        self.assertIsInstance(opt_mod, torch._dynamo.OptimizedModule)

        x = torch.randn(10, 10)
        self.assertTrue(torch._dynamo.testing.same(mod(x), opt_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_attr_precedence(self):
        # [FUNCTION DOC] Test that attribute lookup precedence in a subclass (instance
        # attributes over methods) is respected under compiled execution with module guards
        # enabled: one frame and equal result.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Base module with attribute a, a forward with default c, and
            # methods linear and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets a to three.
                super().__init__()
                self.a = 3

            def forward(self, x, c=4):
                # [FUNCTION DOC] HELPER: Returns x times c.
                return x * c

            def linear(self, x):
                # [FUNCTION DOC] HELPER: Identity method shadowed in the subclass.
                return x

            def b(self, x):
                # [FUNCTION DOC] HELPER: Method that raises RuntimeError if called; shadowed by
                # an instance attribute in the subclass.
                raise RuntimeError("Should not be called")

        class MyMod(Mod):
            # [CLASS DOC] HELPER: Subclass that overrides attributes linear, a, b and scale with
            # instance values.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets linear to a Linear layer, a and b to two, and
                # scale to one.
                super().__init__()
                self.linear = torch.nn.Linear(11, 11)
                self.a = 2
                self.b = 2
                self.scale = 1

            def scale(self, x):
                # [FUNCTION DOC] HELPER: Method shadowed by the instance attribute scale; raises
                # RuntimeError if called.
                # Should not be called because it is shadowed by the instance
                # attribute
                raise RuntimeError("Should not be called")

            def forward(self, x, c=None):
                # [FUNCTION DOC] HELPER: Multiplies linear of x by a, b and scale.
                return self.linear(x) * self.a * self.b * self.scale

        mod = MyMod()
        x = torch.ones(3, 3)
        ref = mod(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnts)
        opt_mod(torch.ones(3, 3))
        res = opt_mod(torch.ones(3, 3))

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref, res)

    def test_to(self):
        # [FUNCTION DOC] Test that calling to on an OptimizedModule keeps it an OptimizedModule,
        # that changing dtype causes one recompile, and that reset forces a further compile;
        # frame counts one, one, two, two, three.
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnt)
        x = torch.randn(10, 10)
        self.assertTrue(torch._dynamo.testing.same(mod(x), opt_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

        # Ensure that there is no recompilation
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 1)

        opt_mod = opt_mod.to(device="cpu").to(dtype=torch.float64)
        self.assertIsInstance(opt_mod, torch._dynamo.OptimizedModule)
        x = torch.randn(10, 10).to(dtype=torch.float64)
        opt_mod(x)
        # Ensure that there is a recompilation
        self.assertEqual(cnt.frame_count, 2)

        # Ensure that there is no recompilation
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 2)

        torch._dynamo.reset()
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 3)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_param_order(self):
        # [FUNCTION DOC] Test with module guards that iterating parameters and named_parameters
        # is guarded on order: reordering the parameter dictionary causes a recompile and
        # results stay correct.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two parameters of different size.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates param1 (size one) and param2 (size two).
                super().__init__()
                self.param1 = torch.nn.Parameter(torch.ones([1]))
                self.param2 = torch.nn.Parameter(torch.ones([2]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x unchanged.
                return x

        mod = MyModule()
        coeffs = [2, 3]

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds weighted parameter sums over parameters and over
            # named_parameters to x.
            for idx, p in enumerate(mod.parameters()):
                x += p.sum() * coeffs[idx]

            for idx, p in enumerate(mod.named_parameters()):
                x += p[1].sum() * coeffs[idx]

            return x

        ref = fn(torch.ones(1))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._parameters["param1"] = mod._parameters.pop("param1")
        ref = fn(torch.ones(1))
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_buffer_order(self):
        # [FUNCTION DOC] Test with module guards that iterating buffers and named_buffers is
        # guarded on order: moving a buffer in the buffers dictionary causes a recompile.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two buffers of different size.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates b1 (size one) and b2 (size two).
                super().__init__()
                self.b1 = torch.nn.Buffer(torch.ones([1]))
                self.b2 = torch.nn.Buffer(torch.ones([2]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x unchanged.
                return x

        mod = MyModule()
        coeffs = [2, 3]

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds weighted buffer sums over buffers and over
            # named_buffers to x.
            for idx, p in enumerate(mod.buffers()):
                x += p.sum() * coeffs[idx]

            for idx, p in enumerate(mod.named_buffers()):
                x += p[1].sum() * coeffs[idx]

            return x

        ref = fn(torch.ones(1))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._buffers["b1"] = mod._buffers.pop("b1")
        ref = fn(torch.ones(1))
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_module_order(self):
        # [FUNCTION DOC] Test with module guards that iterating modules, named_modules, children
        # and named_children is guarded on order: reordering submodules causes a recompile (two
        # frames).
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear children.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear1 and linear2.
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x unchanged.
                return x

        mod = MyModule()
        coeffs = [2, 3, 4]

        coeffs_for_mod = {mod: 10, mod.linear1: 20, mod.linear2: 30}

        # Check order of _modules
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds order-weighted constants for the module and its
            # children, looked up by module identity, over four traversal methods.
            for idx, p in enumerate(mod.modules()):
                # Something silly to force dependency on the order
                x += coeffs_for_mod[p] * coeffs[idx]
            for idx, p in enumerate(mod.named_modules()):
                x += coeffs_for_mod[p[1]] * coeffs[idx]
            for idx, p in enumerate(mod.children()):
                x += coeffs_for_mod[p] * coeffs[idx]
            for idx, p in enumerate(mod.named_children()):
                x += coeffs_for_mod[p[1]] * coeffs[idx]
            return x

        ref = fn(torch.ones(1))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._modules["linear1"] = mod._modules.pop("linear1")
        ref = fn(torch.ones(1))
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_attr(self):
        # [FUNCTION DOC] Test that an OptimizedModule exposes the same parameters and buffers
        # (identical objects) as the original, and that a compiled helper can read parameter
        # dtype.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear and a buffer, whose forward uses an
            # undefined method r.
            # KNOWN ISSUE: forward calls self.r which does not exist; it is never called.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear and a buffer buf0.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.buf0 = torch.nn.Buffer(torch.randn(10, 10))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Would add the buffer to the result of self.r applied to
                # sin of x; never executed.
                return self.r(torch.sin(x)) + self.buf0

        mod = MockModule()
        opt_mod = torch.compile(mod, backend="eager")

        # Check parameters and buffers
        for p1, p2 in zip(mod.parameters(), opt_mod.parameters()):
            self.assertTrue(id(p1) == id(p2))
        for b1, b2 in zip(mod.buffers(), opt_mod.buffers()):
            self.assertTrue(id(b1) == id(b2))

        def get_parameter_dtype(mod: torch.nn.Module):
            # [FUNCTION DOC] HELPER: Returns the dtype of the first item from the chain of
            # parameters and buffers.
            parameters_and_buffers = itertools.chain(mod.parameters(), mod.buffers())
            return next(parameters_and_buffers).dtype

        opt_mod = torch.compile(get_parameter_dtype, backend="eager")
        out_dtype = opt_mod(mod)
        self.assertEqual(out_dtype, torch.float32)

    def test_dir(self):
        # [FUNCTION DOC] Test that dir on an OptimizedModule includes the user attributes,
        # parameter and buffer, and that no attribute in dir of the original is missing.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with Linear, nested buffer and registered parameter;
            # forward is unused.
            # KNOWN ISSUE: forward calls an undefined method r and is never called.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear, buf0 (a Buffer wrapping a Buffer) and
                # param0.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.buf0 = torch.nn.Buffer(torch.nn.Buffer(torch.randn(10, 10)))
                self.register_parameter(
                    name="param0", param=torch.nn.Parameter(torch.randn(10, 10))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Never executed; refers to undefined method r.
                return self.r(torch.sin(x)) + self.buf0

        mod = MockModule()
        mod_keys = dir(mod)
        opt_mod = torch.compile(mod, backend="eager")
        opt_mod_keys = dir(opt_mod)

        # Check user-defined attributes, parameters and buffers
        self.assertIn("linear", opt_mod_keys)
        self.assertIn("buf0", opt_mod_keys)
        self.assertIn("param0", opt_mod_keys)

        # Check all attributes, parameters and buffers
        self.assertTrue(len(set(mod_keys).difference(opt_mod_keys)) == 0)

    def test_no_recompile_on_nn_guarded_modules(self):
        # [FUNCTION DOC] Test that several separately compiled submodules inside one non-
        # compiled parent do not exceed the recompile limit or recompile when error_on_recompile
        # is on: only one frame results.
        size = (10, 10)
        recompile_limit = 1
        num_submodules = 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer applied after sin of cos of the
            # input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear of the module size.
                super().__init__()
                self.linear = torch.nn.Linear(*size)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies cos then sin to the input and then the linear
                # layer.
                a = torch.sin(torch.cos(x))
                return self.linear(a)

        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module holding a list of compiled SubModules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates num_submodules SubModules and replaces each
                # with its compiled version.
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each compiled submodule in sequence.
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # Each submod is compiled separately and has a different nn module
        # guard. Ensure that recompilation logic is handle correctly.
        with (
            unittest.mock.patch("torch._dynamo.config.error_on_recompile", True),
            unittest.mock.patch(
                "torch._dynamo.config.recompile_limit",
                recompile_limit,
            ),
        ):
            x = torch.randn(*size, requires_grad=True)
            mod(x)
            self.assertEqual(cnts.frame_count, 1)

    def test_inline_inbuilt_nn_modules(self):
        # [FUNCTION DOC] Same scenario as the previous nn-guarded-modules test, again checking
        # that separately compiled submodules do not recompile under error_on_recompile; expects
        # one frame.
        size = (10, 10)
        recompile_limit = 1
        num_submodules = 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer applied after sin of cos of the
            # input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear of the module size.
                super().__init__()
                self.linear = torch.nn.Linear(*size)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies cos then sin to the input and then the linear
                # layer.
                a = torch.sin(torch.cos(x))
                return self.linear(a)

        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module holding a list of compiled SubModules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates num_submodules SubModules and compiles each.
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each compiled submodule in sequence.
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # Each submod is compiled separately and has a different nn module
        # guard. Ensure that recompilation logic is handle correctly.
        with (
            unittest.mock.patch("torch._dynamo.config.error_on_recompile", True),
            unittest.mock.patch(
                "torch._dynamo.config.recompile_limit",
                recompile_limit,
            ),
        ):
            x = torch.randn(*size, requires_grad=True)
            mod(x)
            self.assertEqual(cnts.frame_count, 1)

    def test_recompile_limit_on_guarded_nn_modules(self):
        # [FUNCTION DOC] Test that with a recompile limit of two and inputs of three different
        # ranks, the counted frame total ends at two.
        # KNOWN ISSUE: the comment in the test says the expected count is twice the number of
        # submodules, but the assertion checks for two.
        recompile_limit = 2
        num_submodules = 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ReLU applied after sin of cos.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates relu.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies cos then sin to the input, then relu.
                a = torch.sin(torch.cos(x))
                return self.relu(a)

        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module holding a list of compiled SubModules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates compiled submodules.
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each compiled submodule in sequence.
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # For the third iteration, we would reach the cache size limit, and
        # therefore the total number of expected frame count is 2 *
        # num_submodules.
        with unittest.mock.patch(
            "torch._dynamo.config.recompile_limit",
            recompile_limit,
        ):
            for size in [
                (4,),
                (4, 4),
                (4, 4, 4),
            ]:
                x = torch.randn(size)
                mod(x)
        self.assertEqual(cnts.frame_count, 2)

    def test_recursion(self):
        # [FUNCTION DOC] Test that repeatedly compiling an already compiled module six times
        # total still yields a single frame.
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnt)

        for _ in range(5):
            opt_mod = torch.compile(opt_mod, backend=cnt)
        opt_mod(torch.randn(10, 10))
        self.assertEqual(cnt.frame_count, 1)

    def test_composition(self):
        # [FUNCTION DOC] Test that a plain inner module used inside a compiled outer module is
        # traced with one frame, and the outer is an OptimizedModule matching eager.
        class InnerModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying relu to sin of x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates relu.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of sin of x.
                return self.relu(torch.sin(x))

        opt_inner_mod = InnerModule()

        class OuterModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module applying inner to cos of x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the inner module as mod.
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns mod applied to cos of x.
                return self.mod(torch.cos(x))

        outer_mod = OuterModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_outer_mod = torch.compile(outer_mod, backend=cnt)

        x = torch.randn(4)
        self.assertIsInstance(opt_outer_mod, torch._dynamo.OptimizedModule)
        self.assertTrue(torch._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

    def test_composition_with_opt_mod(self):
        # [FUNCTION DOC] Test that an outer module containing an already compiled inner module
        # is compiled with a graph break (two frames).
        class InnerModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying relu to sin of x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates relu.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of sin of x.
                return self.relu(torch.sin(x))

        inner_mod = InnerModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_inner_mod = torch.compile(inner_mod, backend=cnt)

        class OuterModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module applying the compiled inner module to cos of x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the compiled inner module as mod.
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns mod applied to cos of x.
                return self.mod(torch.cos(x))

        outer_mod = OuterModule()
        opt_outer_mod = torch.compile(outer_mod, backend=cnt)

        x = torch.randn(4)
        self.assertIsInstance(opt_outer_mod, torch._dynamo.OptimizedModule)
        self.assertTrue(torch._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        # There will be a graph break for the inner mod being OptimizedModule
        self.assertEqual(cnt.frame_count, 2)

    def test_module_patch(self):
        # [FUNCTION DOC] Test that a forward method taken from ModulePatch2 and bound onto a
        # ModulePatch1 instance is traced with fullgraph; compiled output on a ones tensor must
        # be close to zeros (the patched forward subtracts one).
        mod = ModulePatch1()
        mod.forward = types.MethodType(ModulePatch2.forward, mod)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the patched module applied to x.
            return mod(x)

        self.assertTrue(
            torch.allclose(
                torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(10)),
                torch.zeros(1),
            )
        )

    def test_celu_matches_eager(self):
        # [FUNCTION DOC] Test over alpha values and inplace flags that celu (in place or out of
        # place) compiled with fullgraph gives the same output as eager and mutates the input
        # the same way. Runs each combination as a subtest.
        for alpha, inplace in itertools.product((0.5, 1.0, 2.0), (False, True)):
            with self.subTest(alpha=alpha, inplace=inplace):

                def fn(x):
                    # [FUNCTION DOC] HELPER: Applies celu_ or celu to x with the loop's alpha
                    # depending on the inplace flag.
                    if inplace:
                        return torch.celu_(x, alpha=alpha)
                    return torch.celu(x, alpha=alpha)

                eager_inp = torch.linspace(-2, 2, 17)
                compiled_inp = eager_inp.clone()
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

                self.assertEqual(fn(eager_inp), opt_fn(compiled_inp))
                self.assertEqual(eager_inp, compiled_inp)

    def test_celu_zero_alpha_raises(self):
        # [FUNCTION DOC] Test that celu with alpha zero raises RuntimeError mentioning a
        # ZeroDivisionError both in eager and compiled (fullgraph) mode, for in place and out of
        # place variants.
        for inplace in (False, True):
            with self.subTest(inplace=inplace):

                def fn(x):
                    # [FUNCTION DOC] HELPER: Applies celu_ or celu with alpha zero to x
                    # depending on the inplace flag.
                    if inplace:
                        return torch.celu_(x, alpha=0.0)
                    return torch.celu(x, alpha=0.0)

                with self.assertRaisesRegex(
                    RuntimeError, "ZeroDivisionError: alpha cannot be 0 for CELU"
                ):
                    fn(torch.randn(8))

                with self.assertRaisesRegex(
                    RuntimeError, "ZeroDivisionError: alpha cannot be 0 for CELU"
                ):
                    torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(8))

    def test_partial_monkeypatched_forward_no_recompile(self):
        # [FUNCTION DOC] Test that modules whose forward is replaced by a partial-wrapped hook
        # function (similar to accelerate library hooks) share one compiled graph: four Layer
        # instances run with error_on_recompile enabled after the first, and only one frame is
        # compiled.
        class Hook:
            # [CLASS DOC] HELPER: Hook object with no state; its methods pass args through
            # unchanged.
            def pre_forward(self, module, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Returns the positional and keyword arguments unchanged.
                return args, kwargs

            def post_forward(self, module, output):
                # [FUNCTION DOC] HELPER: Returns the output unchanged.
                return output

        def add_hook_to_module(module):
            # [FUNCTION DOC] HELPER: Saves the module's forward as _old_forward, attaches a Hook
            # as _hf_hook, and replaces forward with a partial-wrapped function preserving the
            # original's metadata.
            old_forward = module.forward
            module._old_forward = old_forward
            module._hf_hook = Hook()

            def new_forward(module, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Replacement forward: runs the hook pre-forward, calls
                # the saved old forward, then returns the hook post-forward result.
                args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
                output = module._old_forward(*args, **kwargs)
                return module._hf_hook.post_forward(module, output)

            module.forward = update_wrapper(partial(new_forward, module), old_forward)

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module containing a Linear projection to which
            # add_hook_to_module is applied at construction.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates q_proj as Linear(4, 4) and installs the forward
                # hook wrapper on it.
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4)
                add_hook_to_module(self.q_proj)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns q_proj applied to x.
                return self.q_proj(x)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")

        @torch.compile(backend=cnt)
        def run(layer, x):
            # [FUNCTION DOC] HELPER: Compiled function calling layer on x.
            return layer(x)

        x = torch.randn(2, 4)
        layers = [Layer() for _ in range(4)]

        self.assertTrue(same(layers[0](x), run(layers[0], x)))
        with patch.object(torch._dynamo.config, "error_on_recompile", True):
            for layer in layers[1:]:
                self.assertTrue(same(layer(x), run(layer, x)))
        self.assertEqual(cnt.frame_count, 1)

    @patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_hooks_outer(self):
        # [FUNCTION DOC] Test that with hook guards enabled, a forward hook that doubles output
        # plus one on a compiled module gives the same result as eager (seven for input one),
        # with no guard failure; after removing the hook the compiled module returns three.
        # Docstring-like summary at the end explains that removing a post-forward hook does not
        # fail a guard.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward doubles the input and adds one.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns two times x plus one.
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Forward hook returning two times the output plus one.
            return 2 * output + 1

        handle = m.register_forward_hook(forward_hook)
        inp = torch.tensor(1.0, requires_grad=True)

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback storing the first failure reason in
            # the enclosing scope.
            nonlocal failure_reason
            failure_reason = failure[0]

        compiled_m = torch._dynamo.optimize(
            guard_fail_fn=guard_fail_fn, backend="eager"
        )(m)

        self.assertEqual(compiled_m(inp), m(inp))
        self.assertEqual(compiled_m(inp).item(), 7)
        self.assertTrue(failure_reason is None)

        # what if we remove our hook? we should recompile?
        handle.remove()
        self.assertEqual(compiled_m(inp), m(inp))
        self.assertEqual(compiled_m(inp).item(), 3)
        # self.assertTrue(failure_reason == "hook")

        """
        Summary:
          - removing a hook doesn't fail a guard, because we weren't compiling the hook
            (at least into the same graph) as forward in the first place! We do correctly
            omit calling the removed hook, but since this hook is a post forward hook,
            the 'RETURN' from forward is breaking the graph.

            Why is 'forward' the entrypoint to an InstructionTranslator, after I changed
            the eval_frame entrypoint to Module.__call__?
        """

    @patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_hooks_inner(self):
        # [FUNCTION DOC] Test with hook guards enabled that a function calling a hooked module
        # compiles into one graph with six ops under aot_eager (result 15); removing the hook
        # triggers a recompile caused by a guard on forward_hooks (result 7, frame count two,
        # ten ops); replacing the hook entry after reset gives 15 then 16 for an altered hook.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward doubles the input and adds one.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns two times x plus one.
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Forward hook returning two times the output plus one.
            return 2 * output + 1

        handle = m.register_forward_hook(forward_hook)

        def outer_func(tensor):
            # [FUNCTION DOC] HELPER: Doubles and adds one to the tensor and then calls the
            # hooked module m.
            x = tensor * 2 + 1
            y = m(x)
            return y

        inp = torch.tensor(1.0, requires_grad=True)

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback storing the first failure reason in
            # the enclosing scope.
            nonlocal failure_reason
            failure_reason = failure[0]

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_func = torch._dynamo.optimize(
            guard_fail_fn=guard_fail_fn,
            backend=cc,
        )(outer_func)

        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)

        # We are compiling 1 big graph for all 3 functions including the hook.
        self.assertEqual(cc.frame_count, 1)
        self.assertEqual(cc.op_count, 6)

        # If we remove the hook, we should recompile
        handle.remove()
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 7)
        self.assertTrue("forward_hooks" in failure_reason)
        self.assertEqual(cc.frame_count, 1 + 1)
        self.assertEqual(cc.op_count, 6 + 4)

        # what if instead of removing, we alter our hook?
        torch._dynamo.reset()
        m = TestModule()
        handle = m.register_forward_hook(forward_hook)
        failure_reason = None
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)

        def new_forward_hook(
            module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Replacement hook returning two times the output plus two.
            return 2 * output + 2

        m._forward_hooks[handle.id] = new_forward_hook
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 16)

    def test_forward_hook_handle_attr_in_hasattr(self):
        # [FUNCTION DOC] Test that registering a forward hook inside forward, guarded by a
        # hasattr check for the hook attribute, compiles with fullgraph and matches the eager
        # output (computed after the hook is already registered).
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with encoder and decoder Linear layers that registers a
            # forward hook on itself the first time forward runs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates encoder (8 to 4) and decoder (4 to 8).
                super().__init__()
                self.encoder = torch.nn.Linear(8, 4)
                self.decoder = torch.nn.Linear(4, 8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Registers a hook that adds one to the output unless the
                # attribute _forward_hook already exists, then returns the decoder applied to
                # the encoder output.
                if not hasattr(self, "_forward_hook"):
                    self._forward_hook = self.register_forward_hook(
                        lambda module, inputs, output: output + 1
                    )
                return self.decoder(self.encoder(x))

        model = TestModule().eval()
        x = torch.randn(2, 8)

        with torch.no_grad():
            model(x)
            expected = model(x)

        compiled_model = torch.compile(model, backend="eager", fullgraph=True)
        with torch.no_grad():
            actual = compiled_model(x)

        self.assertEqual(actual, expected)

    def _forward_hook_test_helper(self, model):
        # [FUNCTION DOC] HELPER: Shared test helper: registers a forward hook on every named
        # submodule that records its inputs in a dict selected by the active record mapping,
        # compiles the model with aot_eager, runs two forward and backward passes compiled and
        # two eager, and prints both recorded layer name key sets. Asserts the compiled and
        # eager recorded key sets match and match the hooked module names.
        forward_handles = {}
        compiled_activations = {}
        eager_activations = {}
        activations = None

        def save_activations(name, mod, inp, out):
            # [FUNCTION DOC] HELPER: Hook that stores the module inputs under the given name in
            # the currently active activations mapping.
            activations[name] = inp

        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(
                partial(save_activations, name)
            )

        compiled_model = torch.compile(model, backend="aot_eager")

        activations = compiled_activations
        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            compiled_activations.clear()
            x = torch.randn((20, 10))
            pred = compiled_model(x)
            loss = pred.sum()
            loss.backward()

        activations = eager_activations
        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            eager_activations.clear()
            x = torch.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()

        print(f"Recorded Layers: {compiled_activations.keys()}\n\n")
        print(f"Expected Layers: {eager_activations.keys()}")

        self.assertTrue(compiled_activations.keys() == eager_activations.keys())
        self.assertTrue(activations.keys() == forward_handles.keys())

    def test_hooks_allowed_modules(self):
        # [FUNCTION DOC] Test that hooks on allowed modules (Linear, ReLU in a Sequential) fire
        # correctly under aot_eager compile via the shared hook helper.
        # this test shouldn't care whether hook guards are enabled or not
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Toy model with a Sequential of two Linear and two ReLU layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds net from layers of sizes ten to ten thousand and
                # ten thousand to five, each followed by ReLU.
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns net applied to x.
                return self.net(x)

        model = ToyModel()
        self._forward_hook_test_helper(model)

    def test_hooks_allowed_modules_compiles(self):
        # [FUNCTION DOC] Test that a compiled model with recording forward hooks on all modules
        # compiles in fullgraph mode to one frame, and the hook fired six times in the last
        # iteration (one per module).
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Toy model with a Sequential of two Linear and two ReLU layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds net from layers of sizes ten to ten thousand and
                # ten thousand to five, each followed by ReLU.
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns net applied to x.
                return self.net(x)

        model = ToyModel()
        activations = []

        def save_activations(mod, inp, out):
            # [FUNCTION DOC] HELPER: Hook appending the module input to the activations list.
            activations.append(inp)

        for module in model.modules():
            module.register_forward_hook(save_activations)

        cnt = torch._dynamo.testing.CompileCounter()
        model = torch.compile(model, backend=cnt, fullgraph=True)
        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            activations.clear()
            x = torch.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()
        self.assertEqual(len(activations), 6)
        self.assertEqual(cnt.frame_count, 1)

    def test_hooks_allowed_modules_compiles_self_contained(self):
        # [FUNCTION DOC] Test that output-modifying forward hooks on all modules compile in
        # fullgraph mode into two frames, an input rank change recompiles to four frames, and a
        # repeat of that shape stays at four.
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Toy model with a Sequential, whose forward multiplies two
            # applications of it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds net from layers of sizes ten to ten thousand and
                # ten thousand to five, each followed by ReLU.
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of net applied to x twice.
                return self.net(x) * self.net(x)

        model = ToyModel()
        forward_handles = {}

        def output_modifying_hook(mod, inp, out):
            # [FUNCTION DOC] HELPER: Hook returning two times the output plus one.
            return 2 * out + 1

        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(output_modifying_hook)

        cnt = torch._dynamo.testing.CompileCounter()

        x = torch.randn((20, 10))
        pred_eager = model(x)
        loss_eager = pred_eager.sum()
        eager_loss_bwd = loss_eager.backward()

        model = torch.compile(model, backend=cnt, fullgraph=True)
        pred = model(x)

        loss = pred.sum()
        loss_bwd = loss.backward()

        self.assertEqual(eager_loss_bwd, loss_bwd)
        self.assertEqual(cnt.frame_count, 2)

        # Ndim change, recompile
        pred = model(torch.randn([10, 10, 10]))
        self.assertEqual(cnt.frame_count, 4)

        # Stable
        pred = model(torch.randn([10, 10, 10]))
        self.assertEqual(cnt.frame_count, 4)

    def test_dunder_call_explicitly(self):
        # [FUNCTION DOC] Test that a module whose forward calls a Linear through the explicit
        # dunder call method still triggers forward hooks under compile, using the shared hook
        # helper.
        # hooks should be triggered if explicit calling `__call__`
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Toy model with one Linear whose forward invokes dunder call
            # explicitly.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear (10 to 10000).
                super().__init__()
                self.linear = torch.nn.Linear(10, 10000)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the result of calling linear through the
                # explicit dunder call.
                return self.linear.__call__(x)

        model = ToyModel()
        self._forward_hook_test_helper(model)

    def test_backward_hooks(self):
        # [FUNCTION DOC] Test that full backward hooks and backward pre-hooks registered on
        # every module (including custom non-allowed modules) are called after compiling with
        # aot_eager and two iterations; the recorded key sets must equal the registered handle
        # names.
        # this test shouldn't care whether hook guards are enabled or not

        class CustomLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random weight parameter and matrix multiply
            # forward.
            # not an 'allowed module', so should not graph-break
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Creates weight of shape a by b.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(a, b))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x and weight.
                return torch.mm(x, self.weight)

        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Toy model with a Sequential of three CustomLinear layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds net from CustomLinear layers of 10 to 10, 10 to
                # 10000, 10000 to 5.
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[CustomLinear(10, 10)]
                    + [CustomLinear(10, 10000)]
                    + [CustomLinear(10000, 5)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns net applied to x.
                return self.net(x)

        model = ToyModel()
        backward_hook_handles = {}
        pre_backward_hook_handles = {}

        grad_sizes = {}

        def backward_hook(name, mod, grad_inp, grad_out):
            # [FUNCTION DOC] HELPER: Full backward hook that records generators of input and
            # output gradient shapes under the module name; returns None.
            grad_sizes[name] = (
                (gi.shape for gi in grad_inp),
                (go.shape for go in grad_out),
            )
            return None

        pre_grad_sizes = {}

        def backward_pre_hook(name, mod, grad_out):
            # [FUNCTION DOC] HELPER: Backward pre-hook that records a generator of output
            # gradient shapes under the module name; returns None.
            pre_grad_sizes[name] = (go.shape for go in grad_out)
            return None

        for name, module in model.named_modules():
            backward_hook_handles[name] = module.register_full_backward_hook(
                partial(backward_hook, name)
            )

            pre_backward_hook_handles[name] = module.register_full_backward_pre_hook(
                partial(backward_pre_hook, name)
            )

        model = torch.compile(model, backend="aot_eager")

        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            x = torch.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()

        self.assertTrue(grad_sizes.keys() == backward_hook_handles.keys())
        self.assertTrue(pre_grad_sizes.keys() == pre_backward_hook_handles.keys())

    def test_udo_instance_method_as_hook(self):
        # [FUNCTION DOC] Test that an instance method of a user-defined object registered as a
        # forward pre-hook with prepend and kwargs support is applied when compiling the object
        # itself (fullgraph, eager): output equals the square of input plus one.
        class CustomClass:
            # [CLASS DOC] HELPER: Object that wraps a module and registers its own method as a
            # forward pre-hook.
            def __init__(self, module):
                # [FUNCTION DOC] HELPER: Stores the module and registers func1 as a prepended
                # forward pre-hook with kwargs support; stores the handle.
                self.module = module
                self.handle = self.module.register_forward_pre_hook(
                    self.func1, prepend=True, with_kwargs=True
                )

            def func1(self, module, args, kwargs):
                # [FUNCTION DOC] HELPER: Pre-hook that adds one to the first positional argument
                # and returns the new args tuple and the unchanged kwargs.
                return (args[0] + 1,), kwargs

            def __call__(self, x):
                # [FUNCTION DOC] HELPER: Calls the wrapped module on x.
                return self.module(x)

        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that squares its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Runs only the parent constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x times x.
                return x * x

        model = ToyModel()
        x = torch.zeros((3, 4))
        obj = CustomClass(model)
        out = torch.compile(obj, fullgraph=True, backend="eager")(x)
        self.assertEqual(out, (x + 1) * (x + 1))

    def test_module_dict_iter_name(self):
        # [FUNCTION DOC] Test that iterating a ModuleDict by key in forward and indexing it
        # compiles into one frame and matches eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ModuleDict of a LeakyReLU and a PReLU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates activations from a list of name-module pairs.
                super().__init__()
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each activation selected by iterating the dict
                # keys.
                for activation_name in self.activations:
                    x = self.activations[activation_name](x)
                return x

        cnt = torch._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(torch.ones(10, 10))

        # Compile
        optim_res = torch.compile(MyModule(), backend=cnt)(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_specialized_module___iter__(self):
        # [FUNCTION DOC] Test that taking an iterator of a ModuleList with force_dynamic turned
        # off and calling next on it works with fullgraph compilation and matches eager.
        ml = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
            ]
        )
        ml.torchdynamo_force_dynamic = False

        def f(x):
            # [FUNCTION DOC] HELPER: Gets an iterator of the module list, takes its first layer
            # and applies it to x.
            it = ml.__iter__()
            return next(it)(x)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(10)
        self.assertEqual(f(x), opt_f(x))

    def test_module_dict_iter_keys(self):
        # [FUNCTION DOC] Test that iterating ModuleDict keys to look up activations in forward
        # compiles into one frame and matches eager (same body as the iter_name test).
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ModuleDict of a LeakyReLU and a PReLU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates activations from a list of name-module pairs.
                super().__init__()
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each activation selected by iterating the dict
                # keys.
                for activation_name in self.activations:
                    x = self.activations[activation_name](x)
                return x

        cnt = torch._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(torch.ones(10, 10))

        # Compile
        optim_res = torch.compile(MyModule(), backend=cnt)(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_module_setattr(self):
        # [FUNCTION DOC] Test that setting a plain attribute on a submodule of a Sequential
        # inside a compiled fullgraph function takes effect on the original object.
        models = torch.nn.Sequential(torch.nn.Linear(3, 3))
        models[0].abc = False

        def run():
            # [FUNCTION DOC] HELPER: Sets abc on the first layer to True and runs the Sequential
            # on a random input.
            models[0].abc = True
            x = torch.randn(1, 3)
            return models(x)

        run = torch.compile(run, fullgraph=True, backend="eager")
        run()
        self.assertTrue(models[0].abc)

    def test_module_dict_iter_values(self):
        # [FUNCTION DOC] Test that iterating over ModuleDict values in forward compiles into one
        # frame and matches eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ModuleDict of a LeakyReLU and a PReLU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates activations from a list of name-module pairs.
                super().__init__()
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each module returned by values in order.
                for activation in self.activations.values():
                    x = activation(x)
                return x

        cnt = torch._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(torch.ones(10, 10))

        # Compile
        optim_res = torch.compile(MyModule(), backend=cnt)(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_unspecialized_seq(self):
        # [FUNCTION DOC] Test that setting the training flag on a submodule of a Sequential
        # inside a compiled function gives the same result as eager.
        models = torch.nn.Sequential(torch.nn.Linear(3, 3))

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets training of the first layer to False and runs the
            # Sequential on x.
            models[0].training = False
            return models(x)

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(1, 3)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_no_op_assignment(self):
        # [FUNCTION DOC] Test that a module reassigning a buffer to a casted copy of itself is
        # traced such that the compile graph has no buffers as attributes (counted via a custom
        # backend), and again after dtype change; a subclass with a custom setattr is compiled
        # and the compile count is at least two.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a plain tensor attribute buffer that is reassigned
            # in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates buffer as a random tensor of length four.
                super().__init__()
                self.buffer = torch.rand([4])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to x, reassigns buffer to itself converted
                # like x, and returns buffer plus x.
                # should be a no-op, but causes dynamo to lose the static input
                x = x + 1
                self.buffer = self.buffer.to(x)
                return self.buffer + x

        compiles_without_buffers = 0

        def debug_compile(gm, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Backend that counts graph modules that have zero buffers
            # and returns the module unchanged.
            nonlocal compiles_without_buffers
            compiles_without_buffers += len(list(gm.buffers())) == 0
            return gm

        @torch.compile(backend=debug_compile)
        def foo(mod, x):
            # [FUNCTION DOC] HELPER: Compiled function calling the module.
            return mod(x)

        mod = Mod()
        foo(mod, torch.rand([4]))
        self.assertEqual(compiles_without_buffers, 1)

        foo(mod, torch.rand([4], dtype=torch.half))
        self.assertEqual(compiles_without_buffers, 2)

        class Mod2(Mod):
            # [CLASS DOC] HELPER: Subclass of Mod with an overridden setattr that delegates to
            # the parent.
            def __setattr__(self, name, value):
                # [FUNCTION DOC] HELPER: Custom setattr that delegates to the parent
                # implementation.
                return super().__setattr__(name, value)

        foo(Mod2(), torch.rand([4]))
        # causes two compilations, bc unimplemented custom setattr
        self.assertTrue(compiles_without_buffers >= 2)

    def test_unspec_non_inlinable_module(self):
        # [FUNCTION DOC] Test that an UnspecNonInlinableModule compiles with the eager backend
        # and matches eager output.
        mod = UnspecNonInlinableModule()
        opt_fn = torch.compile(mod, backend="eager")
        x = torch.randn(100)
        actual = opt_fn(x)
        expected = mod(x)
        self.assertEqual(actual, expected)

    def test_mark_static_previously_seen_tensor(self):
        # [FUNCTION DOC] Test that a tensor previously seen as an input (and also registered as
        # a module buffer) is still marked static and unguarded in the compiled graph: the
        # placeholder named l_b_ must have static input type unguarded, with exactly one
        # compile.
        # This test verifies that dynamo will mark
        # the buffers/params of a module as static
        # even if this param was previously seen
        # (ex. as a different input)
        num_compiles = 0

        def debug_compiler(gm, _):
            # [FUNCTION DOC] HELPER: Backend that counts compiles, asserts that the placeholder
            # named l_b_ exists and has static input type unguarded in its metadata, and returns
            # the graph module.
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n for n in gm.graph.nodes if n.op == "placeholder" and n.name == "l_b_"
            ]

            self.assertGreater(len(input_nodes), 0)
            for input_node in input_nodes:
                self.assertEqual(
                    input_node.meta["tensor_dict"]["_dynamo_static_input_type"],
                    "unguarded",
                )

            return gm

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering the given tensor as a plain buffer named
            # buf.
            def __init__(self, buf) -> None:
                # [FUNCTION DOC] HELPER: Registers buf as a buffer through register_buffer (a
                # Buffer wrapper would detach, as the comment explains).
                super().__init__()
                # Changing this one to nn.Buffer fails because `nn.Buffer` does a .detach()
                # so the value in self.tx.output.side_effects will no longer evaluate to True
                self.register_buffer("buf", buf)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns buf times x.
                return self.buf * x

        @torch.compile(backend=debug_compiler)
        def fn(x, b, mod):
            # [FUNCTION DOC] HELPER: Compiled function adding one to b and multiplying by the
            # module output.
            z = b + 1
            return z * mod(x)

        buf = torch.ones(2, 2)
        inp = torch.ones(2)
        mod = TestModule(buf)
        fn(inp, buf, mod)
        self.assertEqual(num_compiles, 1)

    def test_mark_static_nn_module_tensor(self):
        # [FUNCTION DOC] Test that a plain tensor attribute on an nn module is marked as static
        # and unguarded: the placeholder named l_mod_buf in the compiled graph has static input
        # type unguarded; one compile.
        # This test verifies that dynamo will mark
        # the nn module tensor attributes as static
        num_compiles = 0

        def debug_compiler(gm, _):
            # [FUNCTION DOC] HELPER: Backend that counts compiles, asserts that the placeholder
            # named l_mod_buf exists with static input type unguarded, and returns the graph
            # module.
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "placeholder" and n.name == "l_mod_buf"
            ]

            self.assertGreater(len(input_nodes), 0)
            for input_node in input_nodes:
                self.assertEqual(
                    input_node.meta["tensor_dict"]["_dynamo_static_input_type"],
                    "unguarded",
                )

            return gm

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a plain tensor attribute buf.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates buf as a two by two ones tensor.
                super().__init__()
                self.buf = torch.ones(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns buf times x.
                return self.buf * x

        mod = TestModule()

        @torch.compile(backend=debug_compiler)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function multiplying x by the module output.
            return x * mod(x)

        inp = torch.ones(2)
        fn(inp)
        self.assertEqual(num_compiles, 1)

    @torch._inductor.config.patch("freezing", True)
    @torch.no_grad()
    def test_mark_static_with_freezing(self):
        # [FUNCTION DOC] Test that with inductor freezing and no_grad, the module buffer becomes
        # an attribute on the graph module (no placeholder named l_b_, exactly one buffer);
        # replacing the buffer with a new tensor triggers a second compile.
        # This test verifies that dynamo will
        # add buffers/params as attributes of the
        # graph w/ guards if freezing is enabled
        num_compiles = 0

        def debug_compiler(gm, _):
            # [FUNCTION DOC] HELPER: Backend that counts compiles, asserts no placeholder named
            # l_b_ and one buffer on the graph module, and returns it.
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n for n in gm.graph.nodes if n.op == "placeholder" and n.name == "l_b_"
            ]
            self.assertEqual(len(input_nodes), 0)
            self.assertEqual(len(list(gm.buffers())), 1)
            return gm

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing the given tensor as a Buffer.
            def __init__(self, buf) -> None:
                # [FUNCTION DOC] HELPER: Wraps buf in a Buffer and stores it.
                super().__init__()
                self.buf = torch.nn.Buffer(buf)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns buf times x.
                return self.buf * x

        @torch.compile(backend=debug_compiler)
        def fn(x, mod):
            # [FUNCTION DOC] HELPER: Compiled function calling the module.
            return mod(x)

        buf = torch.ones(2, 2)
        inp = torch.ones(2)
        mod = TestModule(buf)
        fn(inp, mod)
        self.assertEqual(num_compiles, 1)
        mod.buf = torch.rand_like(buf)
        fn(inp, mod)
        self.assertEqual(num_compiles, 2)

    @patch.object(torch._dynamo.config, "guard_nn_modules", True)
    def test_guard_on_torch_nn_modules(self):
        # [FUNCTION DOC] Test (related to issue 110048) that with module guards on, a module
        # attribute change (multiplier) triggers a recompile: two frames across ten loops with
        # two constant values, three after multiplier changes.
        # https://github.com/pytorch/pytorch/issues/110048

        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with Linear and an integer multiplier attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear and sets multiplier to ten.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.multiplier = 10

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear of x times multiplier.
                return self.linear(x) * self.multiplier

        mod = MockModule()

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def generate(x, c):
            # [FUNCTION DOC] HELPER: Compiled function returning the module output plus c.
            return mod(x) + c

        for _ in range(10):
            generate(torch.randn(10, 10), 0)
            generate(torch.randn(10, 10), 1)
        self.assertEqual(cnt.frame_count, 2)

        # Ensure that modification in user module causes recompile
        mod.multiplier = 11
        generate(torch.randn(10, 10), 0)
        self.assertEqual(cnt.frame_count, 3)

    def test_setattr_on_compiled_module(self):
        # [FUNCTION DOC] Test (issue 114844) that setting an attribute on a compiled
        # (OptimizedModule) wrapper reaches the original module state: a tensor assigned to x
        # before the call is replaced by the first Linear output during forward in both eager
        # and compiled copies, and the two stored values must be equal.
        # https://github.com/pytorch/pytorch/issues/114844

        class ReplayMutation(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear layers that stores the first layer
            # output in attribute x during forward.
            def __init__(self, inp_size, out_size, inner_size):
                # [FUNCTION DOC] HELPER: Creates Linear1 (inp_size to inner_size), Linear2
                # (inner_size to out_size) and sets x to None.
                super().__init__()
                self.Linear1 = torch.nn.Linear(inp_size, inner_size)
                self.Linear2 = torch.nn.Linear(inner_size, out_size)
                self.x = None

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Stores the Linear1 output in x (a side effect) and
                # returns Linear2 applied to it.
                res = self.Linear1(inp)
                self.x = res
                return self.Linear2(res)

        N, D_in, H, inner = 2, 2, 2, 4
        model = ReplayMutation(D_in, H, inner)
        model2 = copy.deepcopy(model)
        input = torch.ones(N, D_in)

        # Keep some intermediate value in model.x
        model.x = torch.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
        model(input)

        compiled_model = torch.compile(model2, backend="eager")
        compiled_model.x = torch.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
        compiled_model(input)

        self.assertEqual(model.x, compiled_model.x)

    def test_delattr_on_compiled_module(self):
        # [FUNCTION DOC] Test that deleting an attribute set on a compiled wrapper module
        # removes it from both the wrapper and the original module.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

        model = Mod()
        compiled_model = torch.compile(model, backend="eager")
        compiled_model.foo = 42
        del compiled_model.foo

        self.assertFalse(hasattr(model, "foo"))
        self.assertFalse(hasattr(compiled_model, "foo"))

    def test_custom_getattr_hasattr_no_recursion(self):
        # [FUNCTION DOC] Test that a module with a custom getattr falling back to a dict of
        # custom attributes compiles with fullgraph and matches eager, including a subclass that
        # supplies a scale through the custom attribute mapping.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear and a custom attribute fallback that
            # checks for a custom_attributes mapping without recursing infinitely.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear (4 to 4).
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: Custom getattr: tries the parent lookup, and on
                # AttributeError returns the entry in custom_attributes if the name is there,
                # otherwise re-raises.
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    if (
                        hasattr(self, "custom_attributes")
                        and name in self.custom_attributes
                    ):
                        return self.custom_attributes[name]
                    raise

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of linear applied to x.
                return self.linear(x).relu()

        mod = Mod()
        x = torch.randn(2, 4)
        compiled_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(x), compiled_mod(x))

        class ModWithAttrs(Mod):
            # [CLASS DOC] HELPER: Subclass with custom_attributes containing a scale of two.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear via the parent and sets
                # custom_attributes to a mapping with scale 2.0.
                super().__init__()
                self.custom_attributes = {"scale": 2.0}

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear of x times the scale attribute resolved
                # via the custom getattr.
                return self.linear(x) * self.scale

        mod2 = ModWithAttrs()
        compiled_mod2 = torch.compile(mod2, backend="eager", fullgraph=True)
        self.assertEqual(mod2(x), compiled_mod2(x))

    def test_lazy_module_custom_getattr_no_recursion(self):
        # [FUNCTION DOC] Test that a lazy module whose class is switched to a subclass with a
        # custom getattr does not recurse infinitely when compiled; the second compiled call
        # multiplies x by the filled weight two.
        class LazyMod(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            # [CLASS DOC] HELPER: Lazy module with an uninitialized weight that is materialized
            # to ones-shaped of last dim filled with two.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates an UninitializedParameter named weight.
                super().__init__()
                self.weight = torch.nn.UninitializedParameter()

            def initialize_parameters(self, x):
                # [FUNCTION DOC] HELPER: Under no_grad, materializes weight to the last input
                # dimension and fills it with 2.0.
                with torch.no_grad():
                    self.weight.materialize((x.shape[-1],))
                    self.weight.fill_(2.0)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x times weight.
                return x * self.weight

        class LazyModWithGetattr(LazyMod):
            # [CLASS DOC] HELPER: Subclass adding the same custom getattr fallback as in the
            # previous test.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: Custom getattr: tries the parent lookup, and on
                # AttributeError returns the entry in custom_attributes if the name is there,
                # otherwise re-raises.
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    if (
                        hasattr(self, "custom_attributes")
                        and name in self.custom_attributes
                    ):
                        return self.custom_attributes[name]
                    raise

        mod = LazyMod()
        mod.__class__ = LazyModWithGetattr
        x = torch.randn(2, 4)
        compiled_mod = torch.compile(mod, backend="eager")
        compiled_mod(x)
        res = compiled_mod(x)
        self.assertEqual(x * 2.0, res)

    def test_globals_change_in_other_file(self):
        # [FUNCTION DOC] Test that global variable updates made through functions in this file
        # and in test_functions inside a compiled fullgraph function are reflected: all three
        # counters become one and the result is three times ones. Saves and restores the globals
        # via addCleanup.
        global _variable, _variable1

        prev_variable = _variable
        prev_variable1 = _variable1
        prev_test_functions_variable = test_functions._variable

        def restore_globals():
            # [FUNCTION DOC] HELPER: Cleanup callback restoring the saved values of the three
            # globals.
            global _variable, _variable1
            _variable = prev_variable
            _variable1 = prev_variable1
            test_functions._variable = prev_test_functions_variable

        self.addCleanup(restore_globals)

        _variable = 0
        _variable1 = 0
        test_functions._variable = 0

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function: calls update_global through
            # test_functions.call in a nested frame, calls test_functions.update_global, and
            # scales x by the sum of the three globals.
            # Let `update_global` get invoked in a nested frame, to make sure
            # Dynamo is properly modelling globals across frames and files.
            test_functions.call(update_global)
            a = test_functions.update_global(x)
            # Ensure that the updated global values are read
            return x * a * (_variable + _variable1 + test_functions._variable)

        res = fn(torch.ones(10))
        self.assertEqual(_variable, 1)
        self.assertEqual(_variable1, 1)
        # Ensure that the reconstructed bytecode updates the global value in the
        # other file.
        self.assertEqual(test_functions._variable, 1)
        self.assertEqual(res, 3 * torch.ones(10))

    @unittest.skipIf(
        "inductor" not in torch._dynamo.list_backends(),
        "inductor backend is not available",
    )
    def test_save_and_load_inductor(self):
        # [FUNCTION DOC] Test (skipped when the inductor backend is not registered) that saving
        # a module compiled with inductor and loading it back gives a working module equal to
        # original and compiled modules, and that after reset the loaded model generates
        # inductor kernels (kernel count greater than zero). Turns on inductor metrics logs and
        # resets the logs at the end; uses a temporary directory.
        torch._logging.set_logs(inductor_metrics=True)
        mod = MockModule()
        opt_mod = torch.compile(mod, backend="inductor")
        inp = torch.randn(10, 10)
        opt_mod(inp)

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(opt_mod, os.path.join(tmpdirname, "model.pt"))
            # weights_only=False as this is a legacy use case that loads a module
            loaded_model = torch.load(
                os.path.join(tmpdirname, "model.pt"), weights_only=False
            )
        loaded_model(inp)
        self.assertTrue(same_two_models(loaded_model, mod, [inp]))
        self.assertTrue(same_two_models(loaded_model, opt_mod, [inp]))

        torch._dynamo.reset()  # force recompiles
        torch._inductor.metrics.generated_kernel_count = 0
        loaded_model(inp)
        self.assertGreater(torch._inductor.metrics.generated_kernel_count, 0)
        torch._logging.set_logs()

    def test_save_and_load_all_backends(self):
        # [FUNCTION DOC] Test that for every registered backend, saving and loading a compiled
        # module and running both after reset yields the same indicator of whether inductor
        # kernels were generated; backends that raise BackendCompilerFailed are skipped. Writes
        # to a temporary directory and sets metrics logging.
        torch._logging.set_logs(inductor_metrics=True)
        mod = MockModule()
        inp = torch.randn(10, 10)
        for backend in torch._dynamo.list_backends():
            try:
                opt_mod = torch.compile(mod, backend=backend)
                with tempfile.TemporaryDirectory() as tmpdirname:
                    torch.save(opt_mod, os.path.join(tmpdirname, "model.pt"))
                    # weights_only=False as this is a legacy use case that loads a module
                    loaded_model = torch.load(
                        os.path.join(tmpdirname, "model.pt"), weights_only=False
                    )
                torch._dynamo.reset()  # force recompiles
                torch._inductor.metrics.generated_kernel_count = 0
                opt_mod(inp)
                opt_success = torch._inductor.metrics.generated_kernel_count == 0
                torch._dynamo.reset()  # force recompiles
                torch._inductor.metrics.generated_kernel_count = 0
                loaded_model(inp)
                loaded_success = torch._inductor.metrics.generated_kernel_count == 0
                self.assertEqual(opt_success, loaded_success)
            except torch._dynamo.exc.BackendCompilerFailed:
                pass

    torch._logging.set_logs()

    def test_monkeypatching_forward(self):
        # [FUNCTION DOC] Test that replacing the forward on a module instance (using another
        # class forward) after compilation causes a recompile and uses the new forward: frame
        # count one before, two after patching.
        class FakeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with sine forward used as a source of patched forward.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sine of x.
                return torch.sin(x)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with cosine forward that ignores its constructor
            # argument.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Runs the parent constructor only.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the cosine of x.
                return torch.cos(x)

        def helper():
            # [FUNCTION DOC] HELPER: Resets Dynamo, builds the module and compiled wrapper,
            # checks one frame after two runs, patches forward, and checks recompile and equal
            # results.
            torch._dynamo.reset()
            mod = MyModule(3)

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the module on x.
                return mod(x)

            cnt = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnt)
            x = torch.randn(10)

            opt_fn(x)
            opt_fn(x)
            self.assertEqual(cnt.frame_count, 1)

            # Monkeypatch forward
            mod.forward = types.MethodType(FakeModule.forward, mod)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
            self.assertEqual(cnt.frame_count, 2)

        helper()

    def test_monkeypatching_forward_inside_compiled_region(self):
        # [FUNCTION DOC] Test that assigning a function to a module forward inside a fullgraph
        # compiled region works: the module output becomes the patched function result and the
        # assignment persists on the object afterward.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that subtracts one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x minus one.
                return x - 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(mod, x, y):
            # [FUNCTION DOC] HELPER: Compiled function that replaces mod forward with a local
            # function adding y and returns the module call.
            def patch(x):
                # [FUNCTION DOC] HELPER: Local patch function returning x plus y.
                return x + y

            mod.forward = patch
            return mod(x)

        inp0 = torch.ones(3)
        inp1 = torch.ones(3)
        mod = Mod()

        self.assertEqual(fn(mod, inp0, inp1), inp0 + inp1)
        self.assertEqual(mod(inp0), inp0 + inp1)

    def test_user_defined_nn_module_dynamic(self):
        # [FUNCTION DOC] Test that a user subclass of Conv2d with different strides per instance
        # is compiled as three separate frames (the strides are treated as static, not as
        # symbolic ints).
        class Conv2d(torch.nn.Conv2d):
            # [CLASS DOC] HELPER: Conv2d subclass whose forward calls the functional convolution
            # with the layer attributes.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the Conv2d constructor.
                super().__init__(*args, **kwargs)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the functional convolution of x with weight,
                # bias, stride, padding, dilation, and groups.
                x = torch.nn.functional.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                return x

        cnts = torch._dynamo.testing.CompileCounter()
        mod1 = Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
        mod2 = Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
        mod3 = Conv2d(64, 64, kernel_size=(2, 2), stride=(3, 3))

        opt_mod1 = torch.compile(mod1, backend=cnts, fullgraph=True)
        opt_mod2 = torch.compile(mod2, backend=cnts, fullgraph=True)
        opt_mod3 = torch.compile(mod3, backend=cnts, fullgraph=True)

        x = torch.randn(1, 64, 64, 64)
        opt_mod1(x)
        opt_mod2(x)
        opt_mod3(x)

        # Must be 3 compilations. If not marked static there would be 2, because strides would be converted to symints.
        self.assertEqual(cnts.frame_count, 3)

    def test_overridden_call(self):
        # [FUNCTION DOC] Test that a module overriding the dunder call method (which multiplies
        # the forward result by five) is respected under fullgraph compilation.
        class OverRiddenCallModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module overriding the dunder call to multiply the forward
            # result by five.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Runs only the parent constructor.
                super().__init__()

            def __call__(self, x):
                # [FUNCTION DOC] HELPER: Returns five times forward applied to x.
                # Overrides the __call__ method of torch.nn.Module
                return 5 * self.forward(x)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x times three.
                return x * 3

        m = OverRiddenCallModule()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the module on x.
            return m(x)

        x = torch.ones(4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_param_requires_grad(self):
        # [FUNCTION DOC] Test that toggling requires_grad of a submodule layer between
        # iterations causes recompiles: three frames across three iterations (alternating freeze
        # and unfreeze), with backward run each time.
        def adjust_model(model):
            # [FUNCTION DOC] HELPER: Freezes layer2 parameters when the model iteration counter
            # is even, otherwise sets them to require grad.
            to_freeze = model.num_iter % 2 == 0
            if to_freeze:
                for param in model.layer2.parameters():
                    param.requires_grad = False
            else:
                for param in model.layer2.parameters():
                    param.requires_grad = True

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear layers and an iteration counter
            # incremented in forward.
            def __init__(self, input_size, hidden_size, output_size):
                # [FUNCTION DOC] HELPER: Creates layer1 and layer2 of hidden size and sets
                # num_iter to zero; input_size and output_size are unused.
                super().__init__()

                self.layer1 = torch.nn.Linear(hidden_size, hidden_size)
                self.layer2 = torch.nn.Linear(hidden_size, hidden_size)

                self.num_iter = 0

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies layer2 to x plus layer1 bias, increments
                # num_iter, and returns the result.
                x = self.layer2(x + self.layer1.bias)

                self.num_iter += 1
                return x

        input_size = 1024
        hidden_size = 1024
        output_size = 1
        num_samples = 2048
        features = torch.randn(num_samples, input_size)

        model = MyModule(input_size, hidden_size, output_size)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt, fullgraph=True)

        for _ in range(3):
            model.zero_grad(True)
            adjust_model(model)
            res = opt_model(features)
            res.sum().backward()

        # Check that we have recompiled twice, which leads to 3 frames
        self.assertEqual(cnt.frame_count, 3)

    @torch._dynamo.config.patch("use_recursive_dict_tags_for_guards", False)
    def test_param_requires_grad_no_recursive_dict_tags(self):
        # [FUNCTION DOC] Test, with recursive dict tags for guards turned off, that flipping
        # requires_grad on a parameter forces a recompile (one frame then two) and that the
        # gradient is only populated when requires_grad is on.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear (4 to 4).
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear applied to x.
                return self.linear(x)

        model = MyModule()
        model.linear.weight.requires_grad_(False)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt, fullgraph=True)

        x = torch.randn(4, 4)
        opt_model(x).sum().backward()
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsNone(model.linear.weight.grad)

        model.linear.weight.requires_grad_(True)
        opt_model(x).sum().backward()
        self.assertEqual(cnt.frame_count, 2)
        self.assertIsNotNone(model.linear.weight.grad)

    @torch._dynamo.config.patch(skip_tensor_guards_with_matching_dict_tags=True)
    @torch._dynamo.config.patch("use_recursive_dict_tags_for_guards", True)
    def test_param_dtype_change_recompiles_with_recursive_dict_tags(self):
        # [FUNCTION DOC] Test with recursive dict tags and skipping tensor guards when dict tags
        # match that changing a parameter dtype of a module forces a new compile of a fresh
        # compile wrapper and the new result has float64 dtype; frame count ends at two.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a scale parameter of length four.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates scale as a random Parameter.
                super().__init__()
                self.scale = torch.nn.Parameter(torch.randn(4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x times scale.
                return x * self.scale

        model = MyModule()
        x = torch.randn(4)

        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(model, backend=cnt, fullgraph=True)

        self.assertTrue(torch._dynamo.testing.same(model(x), compiled(x)))
        self.assertEqual(cnt.frame_count, 1)

        model.to(dtype=torch.float64)

        recompiled = torch.compile(model, backend=cnt, fullgraph=True)
        result = recompiled(x)

        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch._dynamo.testing.same(model(x), result))
        self.assertEqual(cnt.frame_count, 2)

    def test_param_requires_grad_submodule(self):
        # [FUNCTION DOC] Test that changing requires_grad on a parameter of a nested submodule
        # recompiles (frame counts one then two) and the gradient appears only when
        # requires_grad is on.
        class Inner(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper with a Conv2d.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates conv (3 to 16 channels, kernel three, stride
                # one).
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns conv applied to x.
                return self.conv(x)

        class Outer(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper holding an Inner module.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates inner.
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns inner applied to x.
                return self.inner(x)

        model = Outer()
        model.inner.conv.weight.requires_grad_(False)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt, fullgraph=True)

        x = torch.randn(1, 3, 10, 10)
        opt_model(x).sum().backward()
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsNone(model.inner.conv.weight.grad)

        model.inner.conv.weight.requires_grad_(True)
        opt_model(x).sum().backward()
        self.assertEqual(cnt.frame_count, 2)
        self.assertIsNotNone(model.inner.conv.weight.grad)

    def test_branch_on_nn_module_custom_len(self):
        # [FUNCTION DOC] Test that a boolean check on a module with a custom length method is
        # traced: an empty cache is falsy so the result is x plus two, and the length method was
        # called exactly once.
        class Cache(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an empty list key_cache and a counter len_invoked;
            # the custom length counts its calls.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates key_cache as an empty list and len_invoked as
                # zero.
                super().__init__()
                self.key_cache = []
                self.len_invoked = 0

            def __len__(self):
                # [FUNCTION DOC] HELPER: Increments the call counter and returns the size of
                # key_cache.
                self.len_invoked += 1
                return len(self.key_cache)

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function that builds a Cache and branches on its
            # truthiness; returns the result and the cache.
            cache = Cache()
            if cache:
                return x + 1, cache
            return x + 2, cache

        x = torch.ones(1)
        res, cache = f(x)
        self.assertEqual(res, x + 2)
        # Make sure Dynamo actually traced the method.
        self.assertEqual(cache.len_invoked, 1)

    def test_branch_on_nn_module_custom_bool(self):
        # [FUNCTION DOC] Test that a boolean check on a module with a custom bool method is
        # traced: the cache is truthy so the result is x plus one, and the method was called
        # exactly once.
        class Cache(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with key_cache holding one item and a counter
            # bool_invoked; the custom bool counts its calls.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates key_cache with one element and bool_invoked as
                # zero.
                super().__init__()
                self.key_cache = [0]
                self.bool_invoked = 0

            def __bool__(self):
                # [FUNCTION DOC] HELPER: Increments the call counter and returns whether
                # key_cache is non-empty.
                self.bool_invoked += 1
                # __bool__ must return a real bool; use truthiness of cache size
                return len(self.key_cache) > 0

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function that builds a Cache and branches on its
            # truthiness; returns the result and the cache.
            cache = Cache()
            if cache:
                return x + 1, cache
            return x + 2, cache

        x = torch.ones(1)
        res, cache = f(x)
        self.assertEqual(res, x + 1)
        # Make sure Dynamo actually traced the method.
        self.assertEqual(cache.bool_invoked, 1)

    def test_patch_module(self):
        # [FUNCTION DOC] Test that a module built by copying the dict of another module plus a
        # bound method from it (patched onto the instance) can be compiled with fullgraph and
        # matches the original module output.
        def set_attrs_from_orig_model(cls_instance, mod, *func_names):
            # [FUNCTION DOC] HELPER: Copies the instance dict of the original module into the
            # target and also copies the named attributes (bound methods) from it.
            cls_instance.__dict__.update(mod.__dict__)
            if func_names is not None:
                for func in func_names:
                    setattr(cls_instance, func, getattr(mod, func))

        class PatchedMyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that copies state from a given module and uses the
            # copied resolve_input method.
            def __init__(self, mod):
                # [FUNCTION DOC] HELPER: Copies attributes of mod including the resolve_input
                # method.
                super().__init__()
                set_attrs_from_orig_model(self, mod, "resolve_input")

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns resolve_input applied to x.
                x = self.resolve_input(x)
                return x

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Original module with a Linear, resolve_input, and forward both
            # applying the linear layer.
            def __init__(self, input_dim, output_dim):
                # [FUNCTION DOC] HELPER: Creates linear with the given input and output
                # features.
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features=input_dim, out_features=output_dim
                )

            def resolve_input(self, x):
                # [FUNCTION DOC] HELPER: Returns linear applied to x.
                x = self.linear(x)
                return x

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear applied to x.
                x = self.linear(x)
                return x

        module = MyModule(input_dim=1, output_dim=1)
        patched_module = PatchedMyModule(module)
        compiled_module = torch.compile(patched_module, backend="eager", fullgraph=True)

        input_tensor = torch.tensor([1.0], dtype=torch.float)
        ref = module(input_tensor)
        res = compiled_module(input_tensor)
        self.assertEqual(ref, res)

    def test_unhashable_nn_submodule(self):
        # [FUNCTION DOC] Test that a module attribute that is a submodule whose hash method
        # raises TypeError does not break compilation of a forward that does not use it.
        class UnhashableModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose hash method raises TypeError.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Raises TypeError with the message Unhashable module.
                raise TypeError("Unhashable module")

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an unhashable submodule and returning its
            # input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates unhashable_attr.
                super().__init__()
                self.unhashable_attr = UnhashableModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x unchanged.
                return x

        mod = MyModule()
        x = torch.randn(1)
        compiled_mod = torch.compile(mod, backend="eager")
        compiled_mod(x)

    def test_trace_delattr(self):
        # [FUNCTION DOC] Test that forward pre and post hooks which rename, delete and restore a
        # module weight attribute trace correctly with fullgraph, giving the same output as
        # eager, and leave the weight present and the temporary attribute absent afterward.
        TMP_PREFIX = "_tmp_"

        def pre_forward_rename_hook(module: torch.nn.Module, _input: torch.Tensor):
            # [FUNCTION DOC] HELPER: Forward pre-hook: saves weight under a temporary name,
            # deletes weight, and sets weight to the original plus one.
            param_name = "weight"
            original_param = getattr(module, param_name)
            setattr(module, TMP_PREFIX + param_name, original_param)
            new_param = original_param + 1.0
            delattr(module, param_name)
            setattr(module, param_name, new_param)

        def post_forward_restore_hook(
            module: torch.nn.Module, _input: torch.Tensor, _output: torch.Tensor
        ):
            # [FUNCTION DOC] HELPER: Forward hook: restores the original weight under the
            # original name and deletes the temporary attribute.
            param_name = "weight"
            tmp_param_name = TMP_PREFIX + param_name
            original_param = getattr(module, tmp_param_name)
            delattr(module, param_name)
            setattr(module, param_name, original_param)
            delattr(module, tmp_param_name)

        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear (10 to 5).
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear applied to x.
                return self.linear(x)

        torch.manual_seed(0)
        model = SimpleModel()

        model.linear.register_forward_pre_hook(pre_forward_rename_hook)
        model.linear.register_forward_hook(post_forward_restore_hook)

        input_tensor = torch.randn(4, 10)

        eager_output = model(input_tensor)
        if not hasattr(model.linear, "weight"):
            raise AssertionError("Expected model.linear to have weight")
        if hasattr(model.linear, "_tmp_weight"):
            raise AssertionError("Expected model.linear to not have _tmp_weight")

        torch.manual_seed(0)
        model_to_compile = SimpleModel()
        model_to_compile.linear.register_forward_pre_hook(pre_forward_rename_hook)
        model_to_compile.linear.register_forward_hook(post_forward_restore_hook)

        compiled_model = torch.compile(
            model_to_compile, fullgraph=True, backend="eager"
        )
        compiled_output = compiled_model(input_tensor)
        if not hasattr(model.linear, "weight"):
            raise AssertionError("Expected model.linear to have weight")
        if hasattr(compiled_model.linear, "_tmp_weight"):
            raise AssertionError(
                "Expected compiled_model.linear to not have _tmp_weight"
            )
        torch.testing.assert_close(eager_output, compiled_output)

    def test_submodule_forward_hooks_with_kwargs(self):
        # [FUNCTION DOC] Test (repro for issue 170110) that compiling many submodules whose
        # inner linear has a forward hook registered with kwargs does not raise or give
        # different outputs than eager.
        # Repro from https://github.com/pytorch/pytorch/issues/170110
        # Tests the forward_hooks_with_kwargs does not result in error
        # or large number of recompiles by default
        class SimpleLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates linear of the given dimension.
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear applied to x.
                return self.linear(x)

        class NLayerModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleDict of numbered SimpleLinear layers.
            def __init__(self, num_layers, dim):
                # [FUNCTION DOC] HELPER: Creates num_layers SimpleLinear modules keyed by their
                # index as a string.
                super().__init__()
                self.layers = torch.nn.ModuleDict(
                    {str(i): SimpleLinear(dim) for i in range(num_layers)}
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each layer in order.
                for layer in self.layers.values():
                    x = layer(x)
                return x

        def noop_hook(module, args, kwargs, output):
            # [FUNCTION DOC] HELPER: Forward hook with kwargs that does nothing.
            pass

        inp = torch.randn(4, 4)
        model = NLayerModel(num_layers=20, dim=4)
        output_eager = model(inp)

        # Set hooks for compiled layers
        for _, layer in model.layers.named_children():
            layer.linear.register_forward_hook(noop_hook, with_kwargs=True)

        for i, layer in model.layers.named_children():
            model.layers.register_module(
                i, torch.compile(layer, fullgraph=True, backend="eager")
            )

        output = model(inp)
        self.assertEqual(output_eager, output)

    # We cannot skip hook_guards here for correctness - otherwise, we will not
    # treat the compiled subgraphs correctly (i.e, setting to True results in
    # incorrect number of compiled hooks called)
    @patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_submodule_forward_hooks_with_kwargs_complex(self):
        # [FUNCTION DOC] Test with hook guards enabled that compiled layers where every third
        # linear has a no-op hook and every third a hook scaling output by half run the hooks
        # the same number of times as eager (seven) and produce equal outputs. The comment above
        # the test explains skipping hook guards would give an incorrect count.
        class SimpleLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Creates linear of the given dimension.
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear applied to x.
                return self.linear(x)

        class NLayerModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleDict of numbered SimpleLinear layers.
            def __init__(self, num_layers, dim):
                # [FUNCTION DOC] HELPER: Creates num_layers SimpleLinear modules keyed by their
                # index as a string.
                super().__init__()
                self.layers = torch.nn.ModuleDict(
                    {str(i): SimpleLinear(dim) for i in range(num_layers)}
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each layer in order.
                for layer in self.layers.values():
                    x = layer(x)
                return x

        def noop_hook(module, args, kwargs, output):
            # [FUNCTION DOC] HELPER: Forward hook with kwargs that does nothing.
            pass

        # Tensor avoids recompiles - check that this is called
        # the same # of times.  NOTE: when skip_nnmodule_hook_guards
        # is false, these values don't match as we explicitly don't guard
        call_count = torch.zeros(1)

        def scale_output(module, args, kwargs, output):
            # [FUNCTION DOC] HELPER: Forward hook with kwargs that increments a shared call
            # counter tensor and returns the output halved.
            call_count[0] += 1
            return output * 0.5

        inp = torch.randn(4, 4)
        model = NLayerModel(num_layers=20, dim=4)
        for idx, (_, layer) in enumerate(model.layers.named_children()):
            if idx % 3 == 0:
                layer.linear.register_forward_hook(noop_hook, with_kwargs=True)
            elif idx % 3 == 1:
                layer.linear.register_forward_hook(scale_output, with_kwargs=True)

        output_eager = model(inp)
        eager_call_count = call_count.item()
        self.assertEqual(eager_call_count, 7)

        for i, layer in model.layers.named_children():
            model.layers.register_module(
                i, torch.compile(layer, fullgraph=True, backend="eager")
            )

        call_count[0] = 0
        output = model(inp)
        compiled_call_count = call_count.item()

        self.assertEqual(compiled_call_count, eager_call_count)
        self.assertTrue(torch.allclose(output_eager, output))

    @patch.object(torch._dynamo.config, "guard_nn_modules", True)
    def test_dict_insertion_guard_method_func(self):
        # [FUNCTION DOC] Test (guard ordering regression for issue 170429) that a compiled
        # fullgraph call works when a module has a plain function pre-hook and a prepended bound
        # method pre-hook, with module guards on; just checks no error.
        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear (10 to 10).
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear applied to x.
                return self.linear(x)

        def hook_function(module, args):
            # [FUNCTION DOC] HELPER: Pre-hook adding one to the first argument and returning a
            # one-element tuple.
            return (args[0] + 1.0,)

        class HookHelper:
            # [CLASS DOC] HELPER: Holder with a pre-hook method.
            def hook_method(self, module, args):
                # [FUNCTION DOC] HELPER: Pre-hook method adding two to the first argument and
                # returning a one-element tuple.
                return (args[0] + 2.0,)

        model = SimpleModel()
        helper = HookHelper()

        model.register_forward_pre_hook(hook_function)
        model.register_forward_pre_hook(helper.hook_method, prepend=True)

        @torch.compile(fullgraph=True, backend="eager")
        def runner_func(mod, x):
            # [FUNCTION DOC] HELPER: Compiled function calling the module on x.
            return mod(x)

        input_tensor = torch.randn(1, 10)
        # This would error before fixing guard orering on nn.Modules (https://github.com/pytorch/pytorch/issues/170429)
        _ = runner_func(model, input_tensor)

    @patch.object(torch._dynamo.config, "guard_nn_modules", True)
    def test_prepend_forward_pre_hook_after_hook_with_closure(self):
        # [FUNCTION DOC] Test that with module guards, a pre-hook registered with prepend after
        # a closure-created hook is ordered correctly: registered key order has the prepended id
        # first while the plain dictionary key order has it last, and the compiled output equals
        # eager.
        # KNOWN ISSUE: the prepended hook returns a bare tensor instead of a tuple, which is
        # accepted as a single replaced argument.
        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear (10 to 10).
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear applied to x.
                return self.linear(x)

        def make_hook(offset):
            # [FUNCTION DOC] HELPER: Factory creating a pre-hook that adds a fixed offset to the
            # first argument.
            def hook(module, args):
                # [FUNCTION DOC] HELPER: Pre-hook adding the captured offset to the first
                # argument and returning a one-element tuple.
                return (args[0] + offset,)

            return hook

        def prepended_hook(module, args):
            # [FUNCTION DOC] HELPER: Pre-hook adding two to the first argument and returning the
            # bare tensor.
            return args[0] + 2.0

        model = SimpleModel()
        existing_handle = model.linear.register_forward_pre_hook(make_hook(1.0))
        prepended_handle = model.linear.register_forward_pre_hook(
            prepended_hook, prepend=True
        )

        self.assertEqual(
            list(model.linear._forward_pre_hooks.keys()),
            [prepended_handle.id, existing_handle.id],
        )
        self.assertEqual(
            list(dict.keys(model.linear._forward_pre_hooks)),
            [existing_handle.id, prepended_handle.id],
        )

        compiled_model = torch.compile(model, fullgraph=True, backend="eager")
        input_tensor = torch.randn(1, 10)
        self.assertEqual(compiled_model(input_tensor), model(input_tensor))

    def test_prepend_hook_ordering(self):
        # [FUNCTION DOC] Test that a Linear subclass with a regular and a prepended pre-hook
        # (static methods) applies them in the right order (first multiply by two, then add one)
        # under fullgraph, matching eager.
        class HookedLinear(torch.nn.Linear):
            # [CLASS DOC] HELPER: Linear subclass registering an add-one pre-hook and a
            # prepended multiply-by-two pre-hook.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards arguments to Linear and registers both hooks.
                super().__init__(*args, **kwargs)
                self.register_forward_pre_hook(self._hook_add)
                self.register_forward_pre_hook(self._hook_mul, prepend=True)

            @staticmethod
            def _hook_add(module, args):
                # [FUNCTION DOC] HELPER: Pre-hook adding one to the first argument.
                return (args[0] + 1,)

            @staticmethod
            def _hook_mul(module, args):
                # [FUNCTION DOC] HELPER: Pre-hook multiplying the first argument by two.
                return (args[0] * 2,)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper holding one HookedLinear.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates layer as a HookedLinear without bias.
                super().__init__()
                self.layer = HookedLinear(4, 4, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns layer applied to x.
                return self.layer(x)

        model = Model()
        x = torch.ones(1, 4)

        eager = model(x)
        compiled = torch.compile(model, backend="eager", fullgraph=True)(x)
        self.assertEqual(eager, compiled)


instantiate_device_type_tests(
    NNModuleTestsDevice, globals(), except_for="cpu", allow_xpu=True
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
