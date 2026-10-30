# Owner(s): ["module: nn"]
# [MODULE DOC] Tests for Module.load_state_dict: error handling, strict versus non-strict
# loading, backward compatible versions, custom save/load hooks, assign mode with meta tensors,
# and parameter swap behavior for tensor subclasses. Defines tensor subclasses (MyLoadTensor,
# MyLoadTensor2, MyBrokenLoadTensor, MyWrapperLoadTensor) with a shared torch function handler
# to emulate the module_load protocol. Parametrized tests are instantiated at import time;
# numpy is imported only if available.
import re
import unittest
from copy import deepcopy
from itertools import product

import torch
import torch.nn as nn
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    swap,
    TEST_NUMPY,
    TestCase,
)
from torch.utils._pytree import tree_map


if TEST_NUMPY:
    import numpy as np


class TestLoadStateDict(NNTestCase):
    # [CLASS DOC] Generic-hardware NNTestCase with CUDA memory leak and non-default stream
    # checks enabled. Exercises load_state_dict on small modules; most tests are run both with
    # and without swap of module params on conversion.
    hw_classification = HardwareClassification.GENERIC
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    @swap([True, False])
    def test_load_state_dict_invalid(self):
        # [FUNCTION DOC] Checks load_state_dict raises a RuntimeError saying a tensor or
        # tensor-like object was expected when a state dict value is a numpy array or a nested
        # tuple of floats. Skipped without numpy; run in both swap modes.
        m = torch.nn.Linear(2, 2, bias=False)

        state_dict = {"weight": np.random.randn(2, 2)}
        with self.assertRaisesRegex(
            RuntimeError,
            "expected torch.Tensor or Tensor-like object from checkpoint but received",
        ):
            m.load_state_dict(state_dict)

        state_dict = {"weight": ((1.0, 1.0), (2.0, 2.0))}
        with self.assertRaisesRegex(
            RuntimeError,
            "expected torch.Tensor or Tensor-like object from checkpoint but received",
        ):
            m.load_state_dict(state_dict)

    @swap([True, False])
    def test_load_state_dict_type(self):
        # [FUNCTION DOC] Checks load_state_dict raises a TypeError saying the state_dict must
        # be dict-like when given an empty string or an integer.
        m = nn.Module()

        with self.assertRaisesRegex(
            TypeError, "Expected state_dict to be dict-like, got"
        ):
            m.load_state_dict("")
        with self.assertRaisesRegex(
            TypeError, "Expected state_dict to be dict-like, got"
        ):
            m.load_state_dict(2)

    @swap([True, False])
    @skipIfTorchDynamo("dynamo installs weakrefs on some params")
    def test_scalar_param_1d_tensor_raises(self):
        # [FUNCTION DOC] Checks a module with a zero-dim parameter: loading a length 3 tensor
        # raises a RuntimeError about a size mismatch for that parameter, while loading a one
        # element 1-d tensor is accepted for backward compatibility and sets the value. Skipped
        # under dynamo.
        class SimpleModule(nn.Module):
            # [CLASS DOC] HELPER: Module with a single scalar parameter named threshold;
            # forward returns its input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the scalar threshold parameter initialized to
                # zero.
                super().__init__()
                self.threshold = nn.Parameter(torch.tensor(0.0))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return x

        m = SimpleModule()

        # Test that [3] -> scalar raises error
        sd = {"threshold": torch.randn(3)}
        with self.assertRaisesRegex(RuntimeError, "size mismatch for threshold"):
            m.load_state_dict(sd)

        # Test that [1] -> scalar is allowed (backward compatibility)
        sd = {"threshold": torch.tensor([1.0])}
        m.load_state_dict(sd)
        self.assertEqual(m.threshold.item(), 1.0)

    @swap([True, False])
    @skipIfTorchDynamo("dynamo installs weakrefs on some params")
    def test_load_state_dict(self):
        # [FUNCTION DOC] Broad load_state_dict test on a module tree with shared linear layer,
        # convolutions, batch norm and an empty child. Checks loading a modified state dict
        # (and a DDP style one after stripping the module prefix) updates values with no
        # missing or unexpected keys; extra, missing and wrong-sized keys raise RuntimeError in
        # strict mode and are reported in the returned incompatible keys when strict is off
        # (wrong size still raises); partial dicts leave other entries untouched. Skipped under
        # dynamo.
        l = nn.Linear(5, 5)
        block = nn.Module()
        block.conv1 = nn.Conv2d(3, 3, 3, bias=True)
        block.conv2 = nn.Conv2d(3, 3, 3, bias=False)
        net = nn.Module()
        net.linear1 = l
        net.linear2 = l
        net.bn = nn.BatchNorm2d(2)
        net.block = block
        net.add_module("empty", None)
        conv1_bias_dtype = block.conv1.bias.dtype

        state_dict = net.state_dict()
        state_dict.update(
            {
                "linear1.weight": torch.ones(5, 5),
                "block.conv1.bias": torch.arange(1, 4, dtype=conv1_bias_dtype),
                "bn.running_mean": torch.randn(2),
            }
        )
        # Also test if a DDP state_dict can be loaded from a local model.
        ddp_state_dict = net.state_dict()
        ddp_state_dict.update(
            {
                "module.linear1.weight": torch.ones(5, 5),
                "module.block.conv1.bias": torch.arange(1, 4, dtype=conv1_bias_dtype),
                "module.bn.running_mean": torch.randn(2),
            }
        )
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            ddp_state_dict, "module."
        )
        for sd in [state_dict, ddp_state_dict]:
            incompatible_keys = net.load_state_dict(sd)
            self.assertEqual(len(incompatible_keys.missing_keys), 0)
            self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
            self.assertNotIn("Incompatible", str(incompatible_keys))
            self.assertEqual(net.linear1.weight, sd["linear1.weight"])
            self.assertEqual(net.block.conv1.bias, sd["block.conv1.bias"])
            self.assertEqual(net.bn.running_mean, sd["bn.running_mean"])

        state_dict = net.state_dict()
        state_dict.update({"extra": torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn("extra", incompatible_keys.unexpected_keys)
        self.assertIn("Incompatible", str(incompatible_keys))

        state_dict = net.state_dict()
        state_dict.update({"extra.param": torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn("extra.param", incompatible_keys.unexpected_keys)

        state_dict = net.state_dict()
        del state_dict["linear1.weight"]
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 1)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
        self.assertIn("linear1.weight", incompatible_keys.missing_keys)
        state_dict.update({"extra.param": torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 1)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn("linear1.weight", incompatible_keys.missing_keys)
        self.assertIn("extra.param", incompatible_keys.unexpected_keys)

        state_dict = net.state_dict()
        state_dict.update({"bn.running_mean": torch.rand(14, 4)})  # wrong size
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        self.assertRaises(
            RuntimeError, lambda: net.load_state_dict(state_dict, strict=False)
        )

        state_dict = net.state_dict()
        old_state_dict = deepcopy(state_dict)
        state_dict = {
            "linear1.weight": torch.ones(5, 5),
            "block.conv1.bias": torch.arange(1, 4, dtype=conv1_bias_dtype),
            "bn.running_mean": torch.randn(2),
            "nonexistent_key": torch.rand(3),
        }
        net.load_state_dict(state_dict, strict=False)
        self.assertEqual(net.linear1.weight, state_dict["linear1.weight"])
        self.assertEqual(net.block.conv1.bias, state_dict["block.conv1.bias"])
        self.assertEqual(net.bn.running_mean, state_dict["bn.running_mean"])
        new_state_dict = net.state_dict()
        del old_state_dict["linear1.weight"]
        del old_state_dict["block.conv1.bias"]
        del old_state_dict["bn.running_mean"]
        for (
            k,
            v,
        ) in old_state_dict.items():
            self.assertTrue(v.equal(new_state_dict[k]))

    @swap([True, False])
    def test_load_state_dict_BC(self):
        # [FUNCTION DOC] Checks backward compatibility for BatchNorm2d: a state dict lacking
        # the batch counter, with version metadata set to 1 or removed, loads successfully and
        # leaves the counter as a long tensor equal to zero.
        # BatchNormNd
        # Added num_batches_tracked buffer at version 2. For state dict with
        # earlier versions or no versions, it should provide default value of 0.
        bn = nn.BatchNorm2d(3)
        state_dict = bn.state_dict()
        del state_dict["num_batches_tracked"]
        state_dict._metadata[""]["version"] = 1  # version 1
        bn.load_state_dict(state_dict)
        self.assertEqual(bn.num_batches_tracked.dtype, torch.long)
        self.assertEqual(bn.num_batches_tracked.item(), 0)
        del state_dict._metadata[""]["version"]  # no version
        bn.load_state_dict(state_dict)
        self.assertEqual(bn.num_batches_tracked.dtype, torch.long)
        self.assertEqual(bn.num_batches_tracked.item(), 0)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/180632")
    @swap([True, False])
    def test_load_state_dict_child(self):
        # [FUNCTION DOC] Builds a nested Sequential tree (three levels of ten copies of a
        # linear layer), registers a load-state-dict pre-hook on one leaf and loads the model's
        # own state dict strictly; the hook asserts the module receives a state dict with the
        # same number of keys as the module's own, i.e. children only see their own keys.
        # Skipped under dynamo.
        base_module = nn.Linear(1, 1)
        model = base_module
        for _ in range(3):
            model = nn.Sequential(*[deepcopy(model) for _ in range(10)])

        def hook_fn(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            # [FUNCTION DOC] HELPER: Load-state-dict pre-hook that asserts the number of keys
            # in the module's own state dict equals the number of keys passed to the hook.
            # Extra arguments are ignored.
            module_state_dict = module.state_dict()
            self.assertEqual(len(module_state_dict.keys()), len(state_dict.keys()))

        model[0][0].register_load_state_dict_pre_hook(hook_fn)
        model.load_state_dict(model.state_dict(), strict=True)

    # fails swapping as LSTM installs weak references on the parameters
    @swap([False])
    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    def test_load_state_dict_ref_cycle(self):
        # [FUNCTION DOC] Checks that loading a copied LSTM state dict does not create reference
        # cycles: after a garbage collection pass, the second collection must report zero
        # collected objects. Only run without swap; skipped under dynamo.
        # load_state_dict shouldn't cause a reference cycle involving Tensors
        import gc

        m = torch.nn.LSTM(16, 16, bidirectional=True)

        gc.collect()
        m.load_state_dict(deepcopy(m).state_dict())
        refcycles = gc.collect()

        self.assertEqual(refcycles, 0)

    @swap([True, False])
    def test_load_state_dict_custom(self):
        # [FUNCTION DOC] Tests custom _save_to_state_dict and _load_from_state_dict overrides
        # nested in a Sequential: the saved dict has the serialized entry holding the parameter
        # plus one and omits the raw parameter, while submodule weights still appear; loading
        # into a fresh module restores the parameter to 10 and sub weight to 555.
        class CustomState(nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter and a linear submodule that
            # serializes its parameter as a single entry holding the value plus one, and
            # restores it by subtracting one on load.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a one element ones parameter and a 5 by 5
                # linear submodule.
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(1))
                self.sub = torch.nn.Linear(5, 5)

            def _save_to_state_dict(self, destination, prefix, keep_vars):
                # [FUNCTION DOC] HELPER: Writes the parameter data plus one into the
                # destination under the prefixed key serialized; keep_vars is ignored and the
                # submodule is saved by the default machinery.
                destination[prefix + "serialized"] = self.param.data + 1

            def _load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                # [FUNCTION DOC] HELPER: Copies the prefixed serialized entry minus one into
                # the parameter data in place; skips all error reporting and ignores strictness
                # and the key lists.
                # skip some of the error handling
                self.param.data.copy_(state_dict[prefix + "serialized"] - 1)

        # use sequential to verify nesting
        m = nn.Sequential(CustomState())
        with torch.no_grad():
            m[0].param[0] = 10
            m[0].sub.weight[0, 0] = 555
        state_dict = m.state_dict()
        self.assertEqual(state_dict["0.serialized"].item(), 11)
        self.assertIn("0.sub.weight", state_dict)
        self.assertNotIn("0.param", state_dict)
        del m
        mm = nn.Sequential(CustomState())
        self.assertEqual(mm[0].param[0].item(), 1)
        mm.load_state_dict(state_dict)
        self.assertEqual(mm[0].param[0].item(), 10)
        self.assertEqual(mm[0].sub.weight[0, 0].item(), 555)

    @swap([True, False])
    @parametrize("keep_vars", [True, False])
    def test_load_state_dict_assign_meta(self, keep_vars):
        # [FUNCTION DOC] Loads a state dict into a meta device copy of a small network with
        # assign=True, parametrized over keep_vars. Verifies parameters and persistent buffers
        # were assigned (identity checks depend on keep_vars and the swap flag, requires_grad
        # preserved, values equal), ordering of named parameters and buffers is preserved, and
        # outputs equal those of the original network. The local variable named swap shadows
        # the imported decorator.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Network with a linear layer, batch norm and a non-trainable
            # parameter added to the batch-normed output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1, bn and the frozen parameter x.
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.bn = nn.BatchNorm1d(5)
                self.x = nn.Parameter(torch.rand(5), requires_grad=False)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the sum of x and the batch norm of the linear
                # output.
                return self.x + self.bn(self.fc1(input))

        swap = torch.__future__.get_swap_module_params_on_conversion()
        net = MyModule()
        state_dict = net.state_dict(keep_vars=keep_vars)
        for v in state_dict.values():
            v.requires_grad_(False)

        with torch.device("meta"):
            net_meta = MyModule()

        net_meta_state_dict_old = net_meta.state_dict(keep_vars=True)
        net_meta.load_state_dict(state_dict, assign=True)

        # Make sure parameters and persistent buffers were assigned
        net_meta_state_dict = net_meta.state_dict(keep_vars=True)
        for key in state_dict:
            if key in net_meta._parameters:
                if keep_vars and not swap:
                    # state_dict[key] is an nn.Parameter
                    self.assertTrue(state_dict[key] is net_meta_state_dict[key])
                else:
                    if swap:
                        self.assertTrue(
                            net_meta_state_dict[key] is net_meta_state_dict_old[key]
                        )
                    else:
                        # state_dict[key] is not an nn.Parameter so it will be detached when wrapping with a Parameter
                        self.assertTrue(
                            net_meta_state_dict[key] is not net_meta_state_dict_old[key]
                        )
                        self.assertEqual(
                            net_meta_state_dict_old[key].requires_grad,
                            net_meta_state_dict[key].requires_grad,
                        )
                self.assertEqual(
                    net_meta_state_dict_old[key].requires_grad,
                    net_meta_state_dict[key].requires_grad,
                )
                self.assertEqual(state_dict[key], net_meta_state_dict[key])
            elif (
                key in net_meta._buffers
                and key not in net_meta._non_persistent_buffers_set
            ):
                self.assertTrue(state_dict[key] is net_meta_state_dict[key])
                self.assertEqual(state_dict[key], net_meta_state_dict[key])

        # Make sure that ordering of parameters and buffers is preserved
        net_named_parameters = net.named_parameters()
        net_named_buffers = net.named_buffers()
        net_meta_named_parameters = net_meta.named_parameters()
        net_meta_named_buffers = net_meta.named_buffers()

        for (n1, _), (n2, _) in zip(net_named_parameters, net_meta_named_parameters):
            self.assertEqual(n1, n2)

        for (n1, _), (n2, _) in zip(net_named_buffers, net_meta_named_buffers):
            self.assertEqual(n1, n2)

        # Make sure outputs are the same
        t = torch.randn(4, 3)
        out_net = net(t)
        out_net_meta = net_meta(t.clone())

        self.assertEqual(out_net, out_net_meta)

    @swap([True, False])
    def test_load_state_dict_assign_with_optimizer(self):
        # [FUNCTION DOC] Trains a network three Adam steps, copies the model and optimizer
        # state dicts, loads the model state into a meta copy with assign=True, then builds a
        # second optimizer afterwards and loads the optimizer state. Both are stepped for three
        # more iterations and optimizer and model state dicts must match.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Network of a linear layer followed by batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1 and bn.
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.bn = nn.BatchNorm1d(5)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the linear layer then batch norm.
                return self.bn(self.fc1(input))

        net = MyModule()
        opt = torch.optim.Adam(net.parameters(), lr=1000)
        x = torch.randn(4, 3)
        num_iters = 3

        for _ in range(num_iters):
            opt.zero_grad()
            out = net(x)
            out.sum().backward()
            opt.step()

        opt_state_dict = deepcopy(opt.state_dict())
        net_state_dict = deepcopy(net.state_dict())

        with torch.device("meta"):
            net_meta = MyModule()

        net_meta.load_state_dict(net_state_dict, assign=True)
        # must create optimizer only after loading state_dict when assign=True
        opt2 = torch.optim.Adam(net_meta.parameters(), lr=1000)
        opt2.load_state_dict(opt_state_dict)

        y = x.clone()
        for _ in range(num_iters):
            opt.zero_grad()
            out = net(x)
            out.sum().backward()
            opt.step()

            opt2.zero_grad()
            out2 = net_meta(y)
            out2.sum().backward()
            opt2.step()

        self.assertEqual(opt.state_dict(), opt2.state_dict())
        self.assertEqual(net.state_dict(), net_meta.state_dict())

    @swap([True, False])
    def test_load_state_dict_assign_shape_stride(self):
        # [FUNCTION DOC] Checks assign=True loading accepts a weight with different strides (a
        # transposed tensor) but raises a RuntimeError about a size mismatch when the shape
        # differs.
        # Assigned tensor is allowed to have different properties than initial
        # tensor except for shape
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Network of a linear layer followed by batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1 and bn.
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.bn = nn.BatchNorm1d(5)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the linear layer then batch norm.
                return self.bn(self.fc1(input))

        net = MyModule()
        state_dict = net.state_dict()
        # loading should be ok if stride is different
        state_dict["fc1.weight"] = torch.randn(3, 5).transpose(0, 1)
        net2 = MyModule()
        net2.load_state_dict(state_dict, strict=False, assign=True)

        state_dict["fc1.weight"] = torch.randn(2, 4)
        with self.assertRaisesRegex(
            RuntimeError, "size mismatch for fc1.weight: copying a param with shape"
        ):
            net2.load_state_dict(state_dict, strict=False, assign=True)

    @swap([True, False])
    def test_load_state_dict_warn_assign(self):
        # [FUNCTION DOC] Checks that loading a CPU tensor into a meta parameter without assign
        # emits a UserWarning about copying from a non-meta parameter to a meta one.
        with torch.device("meta"):
            m = torch.nn.Linear(3, 5)
        state_dict = m.state_dict()
        state_dict["weight"] = torch.empty_like(state_dict["weight"], device="cpu")
        with self.assertWarnsRegex(
            UserWarning,
            "for weight: copying from a non-meta parameter in the checkpoint to a meta",
        ):
            m.load_state_dict(state_dict)

    @swap([True, False])
    def test_load_state_dict_with_unexpected_key(self):
        # [FUNCTION DOC] Checks unexpected keys: strict loading raises a RuntimeError for a key
        # with a bad suffix and for a key whose prefix is a valid parameter name; non-strict
        # loading reports them in the unexpected keys of the result. Note the non-strict loads
        # reuse the state dict assigned inside the preceding raises block.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single linear submodule and no forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the fc1 linear submodule.
                super().__init__()
                self.fc1 = torch.nn.Linear(5, 10)

        m = MyModule()

        # Unexpected key & strict = True
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            state_dict = m.state_dict()
            state_dict["fc1.bad_suffix"] = torch.randn(5, 10)
            m.load_state_dict(state_dict)

        # Unexpected key & strict = False
        state_dict = m.load_state_dict(state_dict, strict=False)
        self.assertIn("fc1.bad_suffix", state_dict.unexpected_keys)

        # Unexpected key whose prefix matches a valid key & strict = True
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            state_dict = m.state_dict()
            state_dict["fc1.weight.bad_suffix"] = torch.randn(5, 10)
            m.load_state_dict(state_dict)

        # Unexpected key whose prefix matches a valid key & strict = False
        state_dict = m.load_state_dict(state_dict, strict=False)
        self.assertIn("fc1.weight.bad_suffix", state_dict.unexpected_keys)


def load_torch_function_handler(cls, func, types, args=(), kwargs=None):
    # [FUNCTION DOC] HELPER: Torch function handler shared by the test tensor subclasses. For
    # the module_load call it converts a source value into the destination's subclass: with
    # assign it returns a detached source; otherwise it rewraps plain tensors, or unwraps
    # wrapper tensors to their data, and raises AssertionError if the source is not an instance
    # of cls when the destination is not. For other functions it runs them with subclass
    # dispatch disabled and makes detach return an instance of cls.
    kwargs = {} if kwargs is None else kwargs

    def module_load(dest, src, assign=False):
        # [FUNCTION DOC] HELPER: Implements the module_load rule described above for a
        # destination and source; returns the value to store in the module. Raises
        # AssertionError on unexpected source or destination types.
        if isinstance(dest, cls):
            if assign:
                return src.detach()
            else:
                if type(src) is torch.Tensor:
                    return cls(src)
                elif type(src) is cls:
                    return src.detach()
                else:
                    if isinstance(src, MyWrapperLoadTensor):
                        return cls(src._data)
                    return cls(src)
        else:
            if not isinstance(src, cls):
                raise AssertionError(
                    f"Expected isinstance(src, {cls}) but got {type(src)}"
                )
            if not (
                type(dest) is torch.Tensor
                or type(dest) is torch.nn.Parameter
                or issubclass(cls, type(dest))
            ):
                raise AssertionError(
                    f"Expected dest to be Tensor, Parameter, or subclass of {cls}, got {type(dest)}"
                )
            if assign:
                return src.detach()
            else:
                if isinstance(src, MyWrapperLoadTensor):
                    if type(dest) not in {torch.Tensor, torch.nn.Parameter}:
                        return type(dest)(src._data)
                    else:
                        return src._data.detach()
                else:
                    return torch.Tensor(src)

    if func is torch.Tensor.module_load:
        return module_load(*args, **kwargs)
    else:
        with torch._C.DisableTorchFunctionSubclass():
            # detach must return instance of same subclass for nn.Parameter()
            if func == torch.Tensor.detach:
                ret = func(*args, **kwargs)
                if not isinstance(ret, cls):
                    return cls(ret)
                return ret
            return func(*args, **kwargs)


class MyLoadTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Tensor subclass whose torch function dispatch goes to the shared load
    # handler, so module_load keeps this subclass.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Forwards to the shared load handler with this class.
        return load_torch_function_handler(cls, func, types, args, kwargs)


# We use MyLoadTensor2 to test tensor subclass, wrapper tensor subclass
# where neither inherits from each other
class MyLoadTensor2(torch.Tensor):
    # [CLASS DOC] HELPER: Second independent tensor subclass using the shared load handler,
    # unrelated by inheritance to the first one.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Forwards to the shared load handler with this class.
        return load_torch_function_handler(cls, func, types, args, kwargs)


class MyBrokenLoadTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Deliberately broken tensor subclass whose module_load returns the
    # source as is without detaching, which must make loading fail.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: For module_load returns the source argument undetached; for
        # other functions runs them with subclass dispatch disabled and wraps detach results in
        # this class.
        kwargs = {} if kwargs is None else kwargs

        if func is torch.Tensor.module_load:
            # wrong as this doesn't detach!
            return args[1]
        else:
            with torch._C.DisableTorchFunctionSubclass():
                # detach must return instance of same subclass for nn.Parameter()
                if func == torch.Tensor.detach:
                    return cls(func(*args, **kwargs))
                return func(*args, **kwargs)


class MyWrapperLoadTensor(MyLoadTensor):
    # [CLASS DOC] HELPER: Wrapper tensor subclass of MyLoadTensor that holds its real data in
    # an attribute and unwraps and rewraps around dispatch.
    @staticmethod
    def __new__(cls, data: torch.Tensor):
        # [FUNCTION DOC] HELPER: Creates a wrapper subclass instance mirroring the size, dtype,
        # layout, device, requires_grad, strides and storage offset of the data tensor.
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            dtype=data.dtype,
            layout=data.layout,
            device=data.device,
            requires_grad=data.requires_grad,
            strides=data.stride(),
            storage_offset=data.storage_offset(),
        )
        return t

    def __init__(self, data: torch.Tensor):
        # [FUNCTION DOC] HELPER: Stores the wrapped data tensor on the instance.
        self._data = data

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string showing the wrapped data.
        return f"MyWrapperLoadTensor({self._data.__repr__()})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch handler that unwraps wrapper arguments to their data,
        # runs the op, and wraps tensor results back into the wrapper class.
        def unwrap(t):
            # [FUNCTION DOC] HELPER: Returns the wrapped data of a wrapper tensor, else the
            # value unchanged.
            return t._data if isinstance(t, MyWrapperLoadTensor) else t

        def wrap(t):
            # [FUNCTION DOC] HELPER: Wraps tensors into the wrapper class, leaves other values
            # unchanged.
            return MyWrapperLoadTensor(t) if isinstance(t, torch.Tensor) else t

        kwargs = {} if kwargs is None else kwargs
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        return tree_map(wrap, out)


class TestLoadStateDictSwap(TestCase):
    # [CLASS DOC] Generic-hardware test case for load_state_dict with swapped module params and
    # tensor subclasses.
    hw_classification = HardwareClassification.GENERIC

    @skipIfCrossRef
    @skipIfTorchDynamo("Can't swap with dynamo as dynamo installs weakrefs")
    @swap([True])
    @parametrize("assign", [True, False])
    def test_swap_subclass(self, assign):
        # [FUNCTION DOC] Parametrized over assign. For every pair of module and state dict
        # subclasses among none and the test subclasses, loads a linear model with a plain or
        # subclassed weight and buffer and checks values, that the weight is still a Parameter,
        # and the resulting types (the source subclass with assign, otherwise the module's own
        # subclass). The broken subclass must raise a RuntimeError reporting errors loading the
        # state dict for Linear. Skipped under dynamo and cross ref.
        def _create_model(subclass=None):
            # [FUNCTION DOC] HELPER: Builds a bias-free linear layer with a random buffer; if a
            # subclass is given, wraps the weight in a Parameter of it and converts the buffer
            # to it.
            m = torch.nn.Linear(2, 3, bias=False)
            m.buf = torch.nn.Buffer(torch.randn(2, 3))
            if subclass is not None:
                m.weight = torch.nn.Parameter(subclass(m.weight))
                m.buf = subclass(m.buf)
            return m

        def _test(m_subclass=None, sd_subclass=None):
            # [FUNCTION DOC] HELPER: Builds a module and a source state dict with the given
            # subclasses, loads with the outer assign flag, then checks values and the expected
            # weight and buffer types.
            m = _create_model(m_subclass)
            sd = _create_model(sd_subclass).state_dict()
            m.load_state_dict(sd, assign=assign)
            self.assertEqual(m.weight, sd["weight"])
            self.assertEqual(m.buf, sd["buf"])
            self.assertTrue(isinstance(m.weight, torch.nn.Parameter))
            self.assertTrue(not isinstance(m.buf, torch.nn.Parameter))

            weight_type, buf_type = (torch.nn.Parameter, torch.Tensor)
            if assign:
                if sd_subclass is not None:
                    weight_type, buf_type = (sd_subclass, sd_subclass)
            else:
                if m_subclass is not None:
                    weight_type, buf_type = (m_subclass, m_subclass)

            self.assertTrue(type(m.weight) is weight_type)
            self.assertTrue(type(m.buf) is buf_type)

        # (MyLoadTensor, MyWrapperLoadTensor) tests the behavior of (superclass, subclass)
        subclasses = [None, MyLoadTensor, MyLoadTensor2, MyWrapperLoadTensor]
        for m_s, sd_s in product(subclasses, subclasses):
            _test(m_s, sd_s)

        # MyBrokenLoadTensor should error since its module_load doesn't call .detach()
        with self.assertRaisesRegex(
            RuntimeError, re.escape("Error(s) in loading state_dict for Linear:")
        ):
            _test(None, MyBrokenLoadTensor)


instantiate_parametrized_tests(TestLoadStateDict)
instantiate_parametrized_tests(TestLoadStateDictSwap)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
