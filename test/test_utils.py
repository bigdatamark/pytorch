# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]
# [MODULE DOC] Miscellaneous tests for torch.utils helpers: activation checkpointing
# (checkpoint, checkpoint_sequential, device state helpers and device inference), DataLoader
# seeding and batching, collect_env, hipify trie and import, torch._assert, standalone C++
# extension JIT loading, call rendering, device context utilities, compiler checks, traceback
# utilities, try_import, internal clock-rate lookup, the deprecated decorator, fork poisoning
# after lazy device init, and environment variable helpers. Importing defines a small dataset
# mock, registers device-type variants of two test classes (the checkpoint device tests and the
# lazy-init test, accelerators only) and the device utils class, and defines a deprecated
# private function whose public alias is created at import by the decorator.

import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
import types
import unittest
import warnings
from typing import Any, cast

import torch
import torch.nn as nn
import torch.utils.cpp_extension
import torch.utils.data
from torch._utils import try_import
from torch._utils_internal import deprecated
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    instantiate_device_type_tests,
    onlyAccelerator,
    onlyCPU,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
    skipIfRocmVersionAtLeast,
    skipIfTorchDynamo,
    TEST_WITH_ASAN,
)
from torch.utils._device import set_device
from torch.utils._pytree import tree_all_only, tree_any
from torch.utils._traceback import (
    CapturedTraceback,
    format_traceback_short,
    report_compile_source_on_error,
)
from torch.utils.checkpoint import (
    _infer_device_type,
    checkpoint,
    checkpoint_sequential,
    get_device_states,
)
from torch.utils.data import DataLoader


# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

from torch.testing._internal.common_utils import run_tests, TestCase


# mypy: disable-error-code="name-defined"


class RandomDatasetMock(torch.utils.data.Dataset):
    # [CLASS DOC] HELPER: Map-style dataset mock of fixed length 1000 whose items are
    # two-element random tensors drawn from torch and Python random generators, so DataLoader
    # seeding with workers can be checked.
    def __getitem__(self, index):
        # [FUNCTION DOC] HELPER: Returns a two-element float tensor built from one torch random
        # number and one Python random uniform number; the index is ignored.
        return torch.tensor([torch.rand(1).item(), random.uniform(0, 1)])

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the fixed dataset length of 1000.
        return 1000


class TestCheckpoint(TestCase):
    # [CLASS DOC] Tests for torch.utils.checkpoint.checkpoint and checkpoint_sequential on CPU:
    # whether recomputation is triggered, gradient equality with the uncheckpointed model, RNG
    # state handling, non-tensor inputs and outputs, partial gradients, early memory freeing for
    # non-reentrant mode and device inference from nested inputs.
    # This runs checkpoint_sequential on each of the nets in
    # module_lists_to_compare, and compares them against the uncheckpointed model.
    # To compare, it checks outputs as well as input gradients and parameter gradients
    def _check_checkpoint_sequential(
        self,
        model,
        module_lists_to_compare,
        num_chunks,
        input,
        use_reentrant,
    ):
        # [FUNCTION DOC] HELPER: Runs the model once without checkpointing and records output,
        # parameter gradients and input gradient. Then for each candidate module list, runs
        # checkpoint_sequential with the given chunk count and reentrancy mode on a detached
        # copy of the input and asserts output, input gradient and every parameter gradient
        # equal the uncheckpointed ones. The model gradients are zeroed between runs.
        # not checkpointed
        out = model(input)
        out_not_checkpointed = out.detach().clone()
        model.zero_grad()
        out.sum().backward()
        grad_not_checkpointed = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
        }
        input_grad_not_checkpointed = input.grad.detach().clone()
        for model_to_compare in module_lists_to_compare:
            # checkpointed model by passing list of modules
            detached = input.detach()
            detached.requires_grad = True

            # pass list of modules to checkpoint
            out = checkpoint_sequential(
                model_to_compare, num_chunks, detached, use_reentrant=use_reentrant
            )
            out_checkpointed = out.detach().clone()
            model.zero_grad()
            out.sum().backward()
            grad_checkpointed = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
            }
            input_grad_checkpointed = detached.grad.detach().clone()
            # compare outputs as well as the gradients of input and parameters
            self.assertEqual(out_checkpointed, out_not_checkpointed)
            self.assertEqual(input_grad_not_checkpointed, input_grad_checkpointed)
            for name in grad_checkpointed:
                self.assertEqual(grad_checkpointed[name], grad_not_checkpointed[name])

    # Test whether checkpoint is being triggered or not. For this, we check
    # the number of times forward pass happens
    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/97402")
    def test_checkpoint_trigger(self):
        # [FUNCTION DOC] Checks checkpoint_sequential over ten counting modules with two
        # segments, for reentrant and non-reentrant modes: every module runs once in the forward
        # pass; after backward the modules in the first half ran twice (recomputed) while those
        # in the second half, the final uncheckpointed segment, ran once. Skipped under Dynamo.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module that counts its forward calls in a counter attribute
            # and multiplies by a constant tensor, used to observe recomputation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and sets a forward call counter
                # to zero.
                super().__init__()
                self.counter = 0

            def forward(self, input_var):
                # [FUNCTION DOC] HELPER: Increments the call counter and returns the input
                # multiplied by a constant tensor, so that autograd saves a tensor and
                # recomputation is observable.
                self.counter += 1
                # For reentrant, need to have autograd actually
                # pack a tensor to trigger recomp
                ret = input_var * torch.tensor(2.0)
                return ret

        # checkpointed
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                modules = [Net() for _ in range(10)]
                for m in modules:
                    self.assertEqual(m.counter, 0)
                input_var = torch.randn(3, 4, requires_grad=True)
                out = checkpoint_sequential(
                    modules, 2, input_var, use_reentrant=use_reentrant
                )
                for m in modules:
                    self.assertEqual(m.counter, 1)
                out.sum().backward()
                for m in modules[: (len(modules) // 2)]:
                    self.assertEqual(m.counter, 2)
                for m in modules[(len(modules) // 2) :]:
                    self.assertEqual(m.counter, 1)

    def test_checkpoint_valid(self):
        # [FUNCTION DOC] Checks that reentrant checkpoint_sequential cannot be used with a
        # double-backward style gradient (create_graph) and raises RuntimeError saying
        # checkpoint is incompatible, while the non-reentrant mode works and gives the same
        # output and gradients as the plain model.
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
        )

        input_var = torch.randn(1, 100, requires_grad=True)

        # checkpointed
        chunks = 2
        modules = list(model.children())
        out = checkpoint_sequential(modules, chunks, input_var, use_reentrant=True)
        with self.assertRaisesRegex(
            RuntimeError, "torch.utils.checkpoint is incompatible"
        ):
            torch.autograd.grad(
                outputs=[out],
                grad_outputs=[torch.ones(1, 5)],
                inputs=[input_var],
                create_graph=True,
            )
        # works with use_reentrant=False, and grads are the same
        out = model(input_var)
        grads_no_checkpoint = torch.autograd.grad(
            outputs=[out],
            grad_outputs=[torch.ones(1, 5)],
            inputs=[input_var],
            create_graph=True,
        )
        out_checkpoint = checkpoint_sequential(
            modules, chunks, input_var, use_reentrant=False
        )
        # check outputs are the same
        self.assertEqual(out_checkpoint, out)
        grads_checkpoint = torch.autograd.grad(
            outputs=[out_checkpoint],
            grad_outputs=[torch.ones(1, 5)],
            inputs=[input_var],
            create_graph=True,
        )
        self.assertEqual(grads_no_checkpoint, grads_checkpoint)

    def test_checkpoint(self):
        # [FUNCTION DOC] For both reentrancy modes, compares checkpoint_sequential on a
        # six-layer sequential model, given both as a list of children and as the module itself,
        # with the uncheckpointed model through the shared helper.
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                model = nn.Sequential(
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5),
                    nn.ReLU(),
                )

                # Compare uncheckpointed model with its checkpointed counterparts
                # In addition to running checkpoint_sequential on the nn.Sequential
                # instance, we also run the function on the list of functions within
                # the module.
                self._check_checkpoint_sequential(
                    model,
                    [list(model.children()), model],
                    2,
                    torch.randn(1, 100, requires_grad=True),
                    use_reentrant=use_reentrant,
                )

    def test_checkpoint_module_list(self):
        # [FUNCTION DOC] Same comparison as the sequential test but for a custom module holding
        # a ModuleList, passing both the list of its children and the ModuleList itself.
        class ModuleListNet(nn.Module):
            # [CLASS DOC] HELPER: Six-layer linear/relu network stored in a ModuleList whose
            # forward applies the layers in order.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a six-layer stack of linear and relu layers and
                # stores it as a ModuleList attribute.
                super().__init__()
                module_list = [
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5),
                    nn.ReLU(),
                ]
                self.module_list = nn.ModuleList(module_list)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies each layer of the module list in order and
                # returns the result.
                for layer in self.module_list:
                    input = layer(input)
                return input

        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                model = ModuleListNet()

                # Compare uncheckpointed model with its checkpointed counterparts.
                self._check_checkpoint_sequential(
                    model,
                    [list(model.module_list.children()), model.module_list],
                    2,
                    torch.randn(1, 100, requires_grad=True),
                    use_reentrant=use_reentrant,
                )

    def test_checkpoint_sequential_deprecated_multiple_args(self):
        # [FUNCTION DOC] Checks that passing more than one input tensor to checkpoint_sequential
        # raises TypeError, for both reentrancy modes (the loop variable is not actually passed
        # to the call).
        class Two(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes two inputs and returns both.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Identity pass-through of two inputs returned as a pair.
                return a, b

        model = nn.Sequential(Two())
        a = torch.randn(1, 100, requires_grad=True)
        b = torch.randn(1, 100, requires_grad=True)

        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                with self.assertRaises(TypeError):
                    checkpoint_sequential(model, 1, a, b)  # type: ignore[call-arg]

    def test_checkpoint_sequential_deprecated_no_args(self):
        # [FUNCTION DOC] Checks that calling checkpoint_sequential with no input raises
        # TypeError, in a loop over both reentrancy modes (the mode is not passed to the call).
        class Noop(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes no input and does nothing.
            def forward(self):
                # [FUNCTION DOC] HELPER: Takes no input and does nothing.
                pass

        model = nn.Sequential(Noop())
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                with self.assertRaises(TypeError):
                    checkpoint_sequential(model, 1)  # type: ignore[call-arg]

    def test_checkpoint_rng_cpu(self):
        # [FUNCTION DOC] Repeats five times: runs two dropout layers on CPU with and without
        # reentrant checkpointing of the second one after saving the global RNG state and
        # restoring it, and asserts the input gradients are identical, showing the RNG state is
        # preserved on recomputation.
        for _ in range(5):
            inp = torch.randn(20000, device="cpu").requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input):
                # [FUNCTION DOC] HELPER: Applies the second dropout layer to the input.
                return phase2(input)

            state = torch.get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out, use_reentrant=True)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    def test_checkpoint_non_tensor(self):
        # [FUNCTION DOC] Checks reentrant checkpoint accepts a None non-tensor argument and
        # still supports backward.
        def run_fn(tensor1, tensor2):
            # [FUNCTION DOC] HELPER: Returns the first tensor, adding the second if it is
            # not None.
            if tensor2 is None:
                return tensor1
            return tensor1 + tensor2

        input_var = torch.randn(1, 100, requires_grad=True)
        out = checkpoint(run_fn, input_var, None, use_reentrant=True)
        out.sum().backward()

    def test_checkpoint_non_tensor_inputs_outputs(self):
        # [FUNCTION DOC] Runs reentrant checkpoint on a function with tensor and non-tensor
        # inputs and outputs (an int, None, a bool, a string). Asserts all returned values,
        # backward through selected outputs, that a second backward raises RuntimeError, and
        # that gradients equal those of the uncheckpointed call.
        def foo(t1, t2, scale, t3):
            # [FUNCTION DOC] HELPER: Computes two scaled tensor expressions of the inputs and
            # returns them with the scale, None, True, a string and the first input in a tuple.
            t4 = t1 + t2 * t3
            t5 = t1 * t2 + t3
            t4 *= scale
            t5 *= scale
            return scale, t4, None, True, t5, "bar", t1

        t1 = torch.rand(10, requires_grad=True)
        t2 = torch.rand(10, requires_grad=True)
        t3 = torch.rand(10)
        scale = random.randint(0, 10)
        res = checkpoint(foo, t1, t2, scale, t3, use_reentrant=True)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # Validate running backward.
        res[1].sum().backward(retain_graph=True)
        res[4].sum().backward(retain_graph=True)
        res[6].sum().backward()
        with self.assertRaisesRegex(
            RuntimeError, "Trying to backward through the graph a second time"
        ):
            res[6].sum().backward()
        t1_grad = t1.grad
        t2_grad = t2.grad

        # Reset grads, run without checkpoint and validate we receive same grads.
        t1.grad = None
        t2.grad = None
        res = foo(t1, t2, scale, t3)
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertEqual(t1.grad, t1_grad)
        self.assertEqual(t2.grad, t2_grad)

    def test_checkpoint_no_tensors(self):
        # [FUNCTION DOC] Runs reentrant checkpoint on a function whose inputs are all Python
        # floats and ints, and asserts the returned values match direct computation.
        def foo(t1, t2, scale, t3):
            # [FUNCTION DOC] HELPER: Same function as in the non-tensor outputs test, computing
            # two scaled expressions and returning them within a tuple of mixed values.
            t4 = t1 + t2 * t3
            t5 = t1 * t2 + t3
            t4 *= scale
            t5 *= scale
            return scale, t4, None, True, t5, "bar", t1

        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = checkpoint(foo, t1, t2, scale, t3, use_reentrant=True)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

    def test_checkpoint_partial_grad(self):
        # [FUNCTION DOC] Checks reentrant checkpoint where only one input requires grad:
        # backward through the first output works, and a function returning only a non-grad
        # input raises RuntimeError saying no output requires grad.
        def run_fn(tensor1, tensor2):
            # [FUNCTION DOC] HELPER: Returns both inputs unchanged as a pair.
            # tensor 2 is used for other application logic
            return tensor1, tensor2

        input_var = torch.randn(1, 4, requires_grad=True)
        input_var2 = torch.randn(1, 4, requires_grad=False)
        out = checkpoint(run_fn, input_var, input_var2, use_reentrant=True)
        out[0].sum().backward()

        def run_fn2(tensor1, tensor2):
            # [FUNCTION DOC] HELPER: Returns only the first input, ignoring the second.
            return tensor1

        input_var = torch.randn(1, 4, requires_grad=False)
        input_var2 = torch.randn(1, 4, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"none of output has requires_grad=True, this checkpoint\(\) is not necessary",
        ):
            out = checkpoint(run_fn2, input_var, input_var2, use_reentrant=True)
            out.sum().backward()

    @unittest.skipIf(not torch.accelerator.is_available(), "No accelerator")
    def test_checkpointing_without_reentrant_early_free(self):
        # [FUNCTION DOC] Requires an accelerator that reports memory use. Compares the memory
        # allocated at each backward hook with plain autograd and with non-reentrant checkpoint,
        # with and without retaining the graph, and asserts the sequences of memory stats match
        # the plain non-retained run. Skips if the backend cannot track allocated memory.
        _acc = torch.accelerator.current_accelerator()
        if _acc is None:
            self.skipTest("current_accelerator() not supported")

        _device_type = _acc.type

        # Functional check: verify memory tracking actually works on this backend
        test_tensor = torch.zeros(1024, device=_device_type)
        torch.accelerator.synchronize()
        if torch.accelerator.memory_allocated() == 0:
            del test_tensor
            self.skipTest(f"{_device_type} does not support memory_allocated tracking")
        del test_tensor

        def _do_test(fn, should_free):
            # [FUNCTION DOC] HELPER: Runs the given driver on a traced function while collecting
            # the allocated memory seen by hooks registered on intermediate tensors; returns the
            # stats list.
            stats: list[int] = []

            def track(x, idx):
                # [FUNCTION DOC] HELPER: Registers a gradient hook on the tensor which records
                # the accelerator memory at index idx and, after the first, asserts that memory
                # decreased when should_free is set or stayed equal otherwise.
                def hook(_unused):
                    # [FUNCTION DOC] HELPER: Gradient hook: checks hooks run in order,
                    # synchronizes, appends the current allocated memory and compares it to the
                    # previous entry.
                    self.assertEqual(len(stats), idx)
                    torch.accelerator.synchronize()
                    stats.append(torch.accelerator.memory_allocated())
                    if idx > 0:
                        if should_free:
                            self.assertLess(stats[idx], stats[idx - 1])
                        else:
                            self.assertEqual(stats[idx], stats[idx - 1])

                x.register_hook(hook)

            def test_fn(x):
                # [FUNCTION DOC] HELPER: Squares the input four times, tracking the three
                # intermediates with hooks numbered in descending order, and returns the sum.
                x = x**2
                track(x, 2)
                x = x**2
                track(x, 1)
                x = x**2
                track(x, 0)
                x = x**2
                return x.sum()

            fn(test_fn)

            return stats

        x = torch.zeros(10, device=_device_type, requires_grad=True)
        x.grad = torch.zeros_like(x)

        non_retain_stats = _do_test(lambda fn: fn(x).backward(), True)

        _unused_retain_stats = _do_test(
            lambda fn: fn(x).backward(retain_graph=True), False
        )

        checkpoint_non_retain_stats = _do_test(
            lambda fn: checkpoint(fn, x, use_reentrant=False).backward(), True
        )

        checkpoint_retain_stats = _do_test(
            lambda fn: checkpoint(fn, x, use_reentrant=False).backward(
                retain_graph=True
            ),
            True,
        )

        self.assertEqual(non_retain_stats, checkpoint_non_retain_stats)
        self.assertEqual(non_retain_stats, checkpoint_retain_stats)

    def test_infer_device_state_recursive_meta(self):
        # [FUNCTION DOC] Checks the checkpoint device inference helper returns the meta type for
        # a dictionary holding a meta tensor.
        inp = {"foo": torch.rand(10, device="meta")}
        device_type = _infer_device_type(inp)
        self.assertEqual("meta", device_type)


class TestCheckpointDeviceType(TestCase):
    # [CLASS DOC] Device-parametrized checkpoint tests; instantiated for every non-CPU
    # accelerator (including XPU), covering RNG preservation, preserve_rng_state disabled,
    # device state collection for nested inputs across two devices, device type inference with
    # warnings and device context preservation on recompute.
    @onlyAccelerator
    def test_checkpoint_rng_accelerator(self, device):
        # [FUNCTION DOC] Accelerator version of the RNG checkpoint test: with and without
        # reentrant checkpointing of a dropout layer, using the device RNG state saved and
        # restored around the runs, the input gradients must be equal.
        for _ in range(5):
            inp = torch.randn(20000, device=device).requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input):
                # [FUNCTION DOC] HELPER: Applies the second dropout layer to the input.
                return phase2(input)

            state = torch.get_device_module(device).get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out, use_reentrant=True)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.get_device_module(device).set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    @onlyAccelerator
    def test_checkpoint_not_preserve_rng_state_and_without_reentrant(self, device):
        # [FUNCTION DOC] Checks non-reentrant checkpoint with RNG state preservation disabled
        # runs forward and backward on the device without error.
        inp = torch.randn(2, device=device).requires_grad_()
        layer = torch.nn.Dropout()

        def run_fn(input):
            # [FUNCTION DOC] HELPER: Applies the dropout layer to the input.
            return layer(input)

        out = checkpoint(run_fn, inp, use_reentrant=False, preserve_rng_state=False)
        out.sum().backward()

    @onlyAccelerator
    @deviceCountAtLeast(2)
    def test_get_device_states_recursive(self, devices):
        # [FUNCTION DOC] Needs two accelerators. Checks get_device_states on a nested dictionary
        # and list of tensors on devices zero and one returns two device ids, zero and one in
        # order, and two tensor states.
        dev_type = torch.device(devices[0]).type
        inp = {
            "foo": torch.rand(10, device=f"{dev_type}:0"),
            "bar": [torch.rand(10, device=f"{dev_type}:1")],
        }
        device_ids, device_states = get_device_states(inp)
        self.assertEqual(2, len(device_ids))
        self.assertEqual(2, len(device_states))
        self.assertEqual(0, device_ids[0])
        self.assertEqual(1, device_ids[1])
        self.assertTrue(isinstance(device_states[0], torch.Tensor))
        self.assertTrue(isinstance(device_states[1], torch.Tensor))

    @onlyAccelerator
    @deviceCountAtLeast(2)
    def test_infer_device_state_recursive_multi_device(self, devices):
        # [FUNCTION DOC] Needs two accelerators. Checks device type inference for nested inputs:
        # devices zero and one of the same type, and two tensors on the same device, raise no
        # warning and return that type; an accelerator tensor plus a meta tensor warns exactly
        # once with a message naming both device types and the first one, and still returns the
        # accelerator type.
        dev_type = torch.device(devices[0]).type
        inp = {
            "foo": torch.rand(10, device=f"{dev_type}:0"),
            "bar": [torch.rand(10, device=f"{dev_type}:1")],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _device_type = _infer_device_type(inp)
            self.assertEqual(dev_type, _device_type)
        inp = {
            "foo": torch.rand(10, device=f"{dev_type}:0"),
            "bar": [torch.rand(10, device=f"{dev_type}:0")],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _device_type = _infer_device_type(inp)
            self.assertEqual(dev_type, _device_type)
        inp = {
            "foo": torch.rand(10, device=f"{dev_type}:0"),
            "bar": [torch.rand(10, device="meta")],
        }
        with warnings.catch_warnings(record=True) as w:
            _device_type = _infer_device_type(inp)
            self.assertEqual(dev_type, _device_type)
        self.assertEqual(len(w), 1)
        warning_msg = str(w[-1].message)
        self.assertTrue(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices"
            in warning_msg
        )
        self.assertTrue(f"Device types: ['{dev_type}', 'meta']" in warning_msg)
        self.assertTrue(f"first device type: {dev_type}" in warning_msg)

    @onlyAccelerator
    def test_checkpoint_preserves_device_context(self, device):
        # [FUNCTION DOC] Checks non-reentrant checkpoint recomputes inside the same default
        # device context: a function creating a tensor with no device records the device type
        # during the forward and the recomputation, which must equal the context device in both.
        """Test that checkpoint preserves torch.device() context during recomputation."""
        recomputed_devices = []
        dev_type = torch.device(device).type

        def fn_creates_tensor(x):
            # [FUNCTION DOC] HELPER: Creates an uninitialized tensor, records its device in the
            # enclosing list and returns the input scaled by its sum.
            intermediate = torch.empty(4)
            recomputed_devices.append(intermediate.device)
            return x * intermediate.sum()

        x = torch.randn(4, device=device, requires_grad=True)

        with torch.device(dev_type):
            y = checkpoint(fn_creates_tensor, x, use_reentrant=False)
            self.assertEqual(recomputed_devices[-1].type, dev_type)

            y.sum().backward()
            self.assertEqual(len(recomputed_devices), 2)
            self.assertEqual(recomputed_devices[-1].type, dev_type)


instantiate_device_type_tests(
    TestCheckpointDeviceType, globals(), except_for=["cpu"], allow_xpu=True
)


class TestDataLoaderUtils(TestCase):
    # [CLASS DOC] Tests for DataLoader behavior: seeding with worker processes and batch
    # counts with and without dropping the last partial batch; workers use a timeout constant
    # of 300 seconds.
    MAX_TIMEOUT_IN_SECOND = 300

    @unittest.skipIf(TEST_WITH_ASAN, "https://github.com/pytorch/pytorch/issues/84937")
    def test_random_seed(self):
        # [FUNCTION DOC] Seeds torch, builds a shuffled four-worker DataLoader over the random
        # mock dataset and takes the first batch, twice with the same seed, and asserts both
        # batches are equal. Skipped under ASAN.
        def run():
            # [FUNCTION DOC] HELPER: Creates a DataLoader with batch size two, four workers,
            # shuffling and the class timeout, and returns its first batch.
            dataloader = torch.utils.data.DataLoader(
                RandomDatasetMock(),
                batch_size=2,
                num_workers=4,
                shuffle=True,
                timeout=self.MAX_TIMEOUT_IN_SECOND,
            )
            return next(iter(dataloader))

        torch.manual_seed(2018)
        x1 = run()
        torch.manual_seed(2018)
        x2 = run()
        self.assertEqual(x1, x2)

    def test_single_keep(self):
        # [FUNCTION DOC] Single-process loader over a tensor of 5 items with batch size 3
        # keeping the last batch: expects two batches.
        # torch.rand(5, 3, 3, 2) is a Tensor here; technically not a valid input because
        # not a Dataset subclass, but needs to stay working so add ignore's
        # for type checking with mypy
        dataloader: DataLoader = DataLoader(
            torch.rand(5, 3, 3, 2),  # type: ignore[arg-type]
            batch_size=3,
            num_workers=0,
            drop_last=False,
        )
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 2)

    def test_single_drop(self):
        # [FUNCTION DOC] Single-process loader over 5 items with batch size 3 dropping the last
        # batch: expects one batch.
        dataloader: DataLoader = DataLoader(
            torch.rand(5, 3, 3, 2),  # type: ignore[arg-type]
            batch_size=3,
            num_workers=0,
            drop_last=True,
        )
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 1)

    @unittest.skip(
        "FIXME: Intermittent GPU out-of-memory error on Windows and time-out under ASAN"
    )
    def test_multi_keep(self):
        # [FUNCTION DOC] Two-worker loader over 5 items with batch size 3 keeping the last
        # batch: expects two batches. Currently skipped unconditionally (intermittent
        # out-of-memory and time-out issues).
        dataloader: DataLoader = DataLoader(
            torch.rand(5, 3, 3, 2),  # type: ignore[arg-type]
            batch_size=3,
            num_workers=2,
            drop_last=False,
            timeout=self.MAX_TIMEOUT_IN_SECOND,
        )
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 2)

    def test_multi_drop(self):
        # [FUNCTION DOC] Two-worker loader over 5 items with batch size 3 dropping the last
        # batch: expects one batch.
        dataloader: DataLoader = DataLoader(
            torch.rand(5, 3, 3, 2),  # type: ignore[arg-type]
            batch_size=3,
            num_workers=2,
            drop_last=True,
            timeout=self.MAX_TIMEOUT_IN_SECOND,
        )
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 1)


test_dir = os.path.abspath(os.path.dirname(str(__file__)))


from torch.utils.collect_env import get_pretty_env_info


@unittest.skipIf(IS_FBCODE, "runs pip which is not available internally")
class TestCollectEnv(TestCase):
    # [CLASS DOC] Smoke test for the environment report collection; skipped in FBCODE where pip
    # is unavailable.
    def test_smoke(self):
        # [FUNCTION DOC] Calls get_pretty_env_info and asserts the report has at least 17
        # line breaks.
        info_output = get_pretty_env_info()
        self.assertTrue(info_output.count("\n") >= 17)


class TestHipify(TestCase):
    # [CLASS DOC] Import check for the hipify tooling.
    def test_import_hipify(self):
        # [FUNCTION DOC] Imports hipify_python from torch.utils.hipify to confirm the
        # package imports.
        from torch.utils.hipify import hipify_python  # noqa: F401


class TestHipifyTrie(TestCase):
    # [CLASS DOC] Tests for the hipify Trie: word insertion and lookup, character quoting for
    # regular expressions and export of the trie into a single regular expression string. State:
    # self.trie, a fresh Trie per test.
    def setUp(self):
        # [FUNCTION DOC] Runs base setup, imports hipify_python and creates a fresh Trie
        # instance stored on the test.
        super().setUp()
        from torch.utils.hipify import hipify_python

        self.trie = hipify_python.Trie()

    def test_add_and_search_trie(self):
        # [FUNCTION DOC] Adds one word, checks the full word is found but its prefix and an
        # unrelated word are not.
        self.trie.add("banana")
        self.assertTrue(self.trie.search("banana"))
        self.assertFalse(self.trie.search("ban"))
        self.assertFalse(self.trie.search("dog"))

    def test_add_multiple_and_search_trie(self):
        # [FUNCTION DOC] Adds several words and checks each is found while prefixes and
        # unrelated words are not.
        words_to_add = ["banana", "apple", "orange"]
        for word in words_to_add:
            self.trie.add(word)

        for word in words_to_add:
            self.assertTrue(self.trie.search(word))

        for word in ["ban", "dog", "okay", "app"]:
            self.assertFalse(self.trie.search(word))

    def test_quote_escape(self):
        # [FUNCTION DOC] Checks the trie quote method escapes regex special characters (star,
        # bracket, dot, plus, dash) with a backslash and leaves letters unchanged.
        orig_chars = ["*", "[", ".", "+", "a", "z", "-"]
        quoted_strs = ["\\*", "\\[", "\\.", "\\+", "a", "z", "\\-"]
        for i in range(len(orig_chars)):
            self.assertEqual(self.trie.quote(orig_chars[i]), quoted_strs[i])

    def test_export_trie_to_regex(self):
        # [FUNCTION DOC] Adds several CUDA error names and checks the exported regular
        # expression equals the expected grouped alternation sorted by shared prefixes.
        words_to_add = [
            "__CUDACC__",
            "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            "CUDA_ERROR_ARRAY_IS_MAPPED",
            "CUDA_ERROR_NOT_MAPPED",
            "CUDA_ERROR_INVALID_SOURCE",
        ]
        for word in words_to_add:
            self.trie.add(word)
        regex = self.trie.export_to_regex()
        expected_regex = r"(?:CUDA_ERROR_(?:ARRAY_IS_MAPPED|CONTEXT_ALREADY_CURRENT|INVALID_SOURCE|NOT_MAPPED)|__CUDACC__)"
        self.assertEqual(regex, expected_regex)

    def test_prefix_words_export_trie_to_regex(self):
        # [FUNCTION DOC] Checks export when some words are prefixes of others (apple/app,
        # ban/banana): shared tails become optional groups.
        # test case where some nodes have both children and are also leaf nodes.
        words_to_add = ["apple", "app", "ban", "banana"]
        for word in words_to_add:
            self.trie.add(word)
        regex = self.trie.export_to_regex()
        expected_regex = r"(?:app(?:le)?|ban(?:ana)?)"
        self.assertEqual(regex, expected_regex)

    def test_single_export_trie_to_regex(self):
        # [FUNCTION DOC] Checks export of a single word is the word itself with no grouping.
        words_to_add = ["cudaErrorInvalidMemcpyDirection"]
        for word in words_to_add:
            self.trie.add(word)
        regex = self.trie.export_to_regex()
        expected_regex = "cudaErrorInvalidMemcpyDirection"
        self.assertEqual(regex, expected_regex)

    def test_char_export_trie_to_regex(self):
        # [FUNCTION DOC] Checks one character exports as itself and two single characters export
        # as a bracketed character class.
        self.trie.add("a")
        self.assertEqual(self.trie.export_to_regex(), "a")
        self.trie.add("b")
        self.assertEqual(self.trie.export_to_regex(), "[ab]")

    def test_special_char_export_trie_to_regex(self):
        # [FUNCTION DOC] Checks a word containing a regex special character exports with that
        # character escaped.
        self.trie.add(r"c*")
        self.assertEqual(self.trie.export_to_regex(), r"c\*")


class TestAssert(TestCase):
    # [CLASS DOC] Tests for torch._assert in eager and scripted modes.
    def test_assert_true(self):
        # [FUNCTION DOC] Checks torch._assert passes for true bool and one-element true tensor
        # conditions and raises AssertionError with the given message for false ones.
        # verify assertions work as expected
        # bool argument
        torch._assert(True, "foo")
        with self.assertRaisesRegex(AssertionError, "bar"):
            torch._assert(False, "bar")
        # tensor argument
        torch._assert(torch.tensor([True], dtype=torch.bool), "foo")
        with self.assertRaisesRegex(AssertionError, "bar"):
            torch._assert(torch.tensor([False], dtype=torch.bool), "bar")

    def test_assert_scriptable(self):
        # [FUNCTION DOC] Scripts a module using torch._assert on a positive-sum condition;
        # checks it passes for a positive input and raises torch.jit.Error with the message for
        # a false tensor input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Scriptable module whose forward asserts that its input sums to
            # a positive value and then returns the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Asserts the input sum is positive (message foo) and
                # returns the input.
                torch._assert(x.sum() > 0, "foo")
                return x

        m = M()
        # scriptable
        ms = torch.jit.script(m)
        # data can be passed without errors
        x = torch.randn(4, 4).fill_(1.0)
        ms(x)
        with self.assertRaisesRegex(torch.jit.Error, "foo"):
            ms(torch.tensor([False], dtype=torch.bool))


@unittest.skipIf(IS_SANDCASTLE, "cpp_extension is OSS only")
class TestStandaloneCPPJIT(TestCase):
    # [CLASS DOC] Tests building and running a standalone C++ binary through cpp_extension.load;
    # skipped on Sandcastle.
    # The standalone binary cannot resolve librocprofiler-sdk.so.1 in CI.
    @skipIfRocmVersionAtLeast([7, 14])
    def test_load_standalone(self):
        # [FUNCTION DOC] Writes a small C++ program printing an identity matrix into a temporary
        # build directory, builds it with load in standalone mode, checks the returned path
        # (with an exe suffix on Windows) and runs it both with and without a shell, comparing
        # return code and normalized stdout. Always removes the build directory. Skipped on
        # newer ROCm versions where the binary cannot resolve a profiler library.
        build_dir = tempfile.mkdtemp()
        try:
            src_path = os.path.join(build_dir, "main.cpp")
            src = textwrap.dedent(
                """\
                #include <iostream>
                #include <torch/torch.h>
                int main() {
                    auto x = torch::eye(3);
                    std::cout << x << std::endl;
                }
            """
            )
            with open(src_path, "w") as f:
                f.write(src)

            exec_path = torch.utils.cpp_extension.load(
                "standalone_load_test",
                src_path,
                build_directory=build_dir,
                is_python_module=False,
                is_standalone=True,
            )

            ext = ".exe" if IS_WINDOWS else ""
            self.assertEqual(
                exec_path, os.path.join(build_dir, f"standalone_load_test{ext}")
            )

            for shell in [True, False]:
                r = subprocess.run(
                    [exec_path],
                    shell=shell,
                    stdout=subprocess.PIPE,
                )
                self.assertEqual(r.returncode, 0)
                self.assertEqual(
                    # Windows prints "\r\n" for newlines.
                    textwrap.dedent(r.stdout.decode("utf-8")).replace("\r\n", "\n"),
                    textwrap.dedent(
                        """\
                     1  0  0
                     0  1  0
                     0  0  1
                    [ CPUFloatType{3,3} ]
                    """
                    ),
                )

        finally:
            shutil.rmtree(build_dir)


class TestRenderUtils(TestCase):
    # [CLASS DOC] Tests for the torch._utils call rendering helper.
    def test_basic(self):
        # [FUNCTION DOC] Checks render_call output for torch.sum with a 1D and a 2D tensor
        # argument and a dimension keyword, using expected-inline strings that abbreviate
        # tensors with their size.
        self.assertExpectedInline(
            torch._utils.render_call(torch.sum, [torch.randn(100)], {"dim": 0}),
            """torch.sum(tensor([...], size=(100,)), dim=0)""",
        )
        self.assertExpectedInline(
            torch._utils.render_call(torch.sum, [torch.randn(100, 100)], {"dim": 0}),
            """torch.sum(tensor([...], size=(100, 100)), dim=0)""",
        )


class TestDeviceUtils(TestCase):
    # [CLASS DOC] Tests for device context utilities: the device context manager, the set_device
    # decorator for functions and generators, module construction under a device, set and get
    # default device, and a sweep over the OpInfo database under the meta device. Instantiated
    # for all device types.
    def test_basic(self):
        # [FUNCTION DOC] Creates a tensor inside a meta device context and checks its device
        # type and that the context manager yields the meta device.
        with torch.device("meta") as dev:
            x = torch.empty(3, 3)
        self.assertEqual(x.device.type, "meta")
        self.assertEqual(dev, torch.device("meta"))

    def test_decorator(self):
        # [FUNCTION DOC] Checks the set_device decorator makes a function create tensors
        # on meta.
        @set_device("meta")
        def f():
            # [FUNCTION DOC] HELPER: Returns a new 3 by 3 uninitialized tensor.
            return torch.empty(3, 3)

        self.assertEqual(f().device.type, "meta")

    def test_decorator_generator(self):
        # [FUNCTION DOC] Checks the set_device decorator also applies to each step of a
        # generator function.
        @set_device("meta")
        def f():
            # [FUNCTION DOC] HELPER: Generator yielding two 3 by 3 uninitialized tensors.
            yield torch.empty(3, 3)
            yield torch.empty(3, 3)

        r1, r2 = list(f())
        self.assertEqual(r1.device.type, "meta")
        self.assertEqual(r2.device.type, "meta")

    def test_nn_module(self):
        # [FUNCTION DOC] Checks a linear module constructed inside a meta device context has
        # meta parameters.
        with torch.device("meta"):
            m = nn.Linear(40, 50)
        self.assertEqual(m.weight.device.type, "meta")

    def test_set_default_device(self):
        # [FUNCTION DOC] Sets the global default device to meta, creates a tensor and checks it
        # is on meta; always resets the default device to None afterwards.
        try:
            torch.set_default_device("meta")
            r = torch.empty(2, 2)
        finally:
            torch.set_default_device(None)

        self.assertEqual(r.device.type, "meta")

    def test_get_default_device(self):
        # [FUNCTION DOC] Sets the default device to meta and checks the getter reports meta,
        # then resets it to None.
        torch.set_default_device("meta")
        self.assertEqual(torch.get_default_device().type, "meta")
        torch.set_default_device(None)

    @onlyAccelerator
    @deviceCountAtLeast(2)
    def test_get_default_device_more(self, devices):
        # [FUNCTION DOC] Needs two accelerators. Exercises the default device getter against
        # actual tensor device while changing the default by device type string, the current
        # device index, an explicit indexed device, a device context overriding the default, and
        # cpu; resets the default to None in a finally block.
        dev_type = torch.device(devices[0]).type
        try:
            torch.set_default_device(dev_type)
            self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
            torch.set_default_device(None)

            torch.set_default_device(dev_type)
            torch.get_device_module(dev_type).set_device(f"{dev_type}:1")
            self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
            torch.accelerator.set_device_index(1)
            self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
            torch.set_default_device(None)

            torch.set_default_device(f"{dev_type}:1")
            self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
            torch.set_default_device(None)

            torch.set_default_device(f"{dev_type}:1")
            with torch.device(f"{dev_type}:0"):
                self.assertEqual(
                    torch.get_default_device(), torch.device(f"{dev_type}", 0)
                )

            torch.set_default_device("cpu")
            self.assertEqual(torch.get_default_device(), torch.device("cpu"))
            with torch.device(f"{dev_type}:0"):
                self.assertEqual(
                    torch.get_default_device(), torch.device(f"{dev_type}", 0)
                )

            self.assertEqual(torch.get_default_device(), torch.device("cpu"))
        finally:
            torch.set_default_device(None)

    @onlyCPU
    @ops(op_db)
    def test_device_mode_ops(self, device, dtype, op):
        # [FUNCTION DOC] CPU only, over every OpInfo: for samples without tensor arguments
        # anywhere, calls the op under a meta device context (with any explicit device keyword
        # removed) and asserts all tensor outputs are on meta.
        func = op.get_op()
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            # Only test samples which don't have Tensor inputs.  However,
            # we don't test the factory property on OpInfo as it is very,
            # very incomplete
            if tree_any(
                lambda x: isinstance(x, torch.Tensor),
                (sample.input, sample.args, sample.kwargs),
            ):
                continue
            # Many OpInfos will explicitly pass in a device.  DeviceContext
            # will respect device if it is explicitly specified.  To test
            # DeviceContext, we have to remove the device kwarg in this case.
            # NB: Can't pass None to sample_inputs, the function can't
            # handle it.
            kwargs = sample.kwargs.copy()
            kwargs.pop("device", None)
            with torch.device("meta"):
                r = func(sample.input, *sample.args, **kwargs)

            def is_meta_device(x: torch.Tensor) -> bool:
                # [FUNCTION DOC] HELPER: Returns whether the tensor is on the meta device.
                return x.device.type == "meta"

            self.assertTrue(tree_all_only(torch.Tensor, is_meta_device, r))


instantiate_device_type_tests(TestDeviceUtils, globals())


class TestCppExtensionUtils(TestCase):
    # [CLASS DOC] Checks that the compilers used for C++ extensions are acceptable for
    # the platform.
    def test_cpp_compiler_is_ok(self):
        # [FUNCTION DOC] Asserts check_compiler_ok_for_platform accepts the command c++.
        self.assertTrue(torch.utils.cpp_extension.check_compiler_ok_for_platform("c++"))

    def test_cc_compiler_is_ok(self):
        # [FUNCTION DOC] Asserts check_compiler_ok_for_platform accepts the command cc.
        self.assertTrue(torch.utils.cpp_extension.check_compiler_ok_for_platform("cc"))


class TestTraceback(TestCase):
    # [CLASS DOC] Tests for torch.utils._traceback: reporting compile source on error, short
    # formatting and CapturedTraceback extraction and formatting.
    def test_basic(self):
        # [FUNCTION DOC] Executes a small source string defining a function that raises, with
        # the source registered as compile source, runs it under the
        # report_compile_source_on_error context manager and, on the RuntimeError, asserts the
        # formatted traceback includes the marker text from the source string.
        source = """\
def f(x):
    def g(x):
        raise RuntimeError  # HEYA

    x = x * 3
    return g(x) + 1
"""

        out: dict[str, Any] = {}
        scope = {"__compile_source__": source}
        exec(source, scope, out)

        try:
            with report_compile_source_on_error():
                out["f"](1)
        except RuntimeError as e:
            self.assertIn("HEYA", "".join(traceback.format_tb(e.__traceback__)))

    def test_format_traceback_short(self):
        # [FUNCTION DOC] Raises and catches an error and asserts format_traceback_short output
        # matches a pattern naming this test file, a line number and this test function.
        try:
            raise RuntimeError
        except RuntimeError as e:
            self.assertRegex(
                format_traceback_short(e.__traceback__),
                r".*test_utils.py:\d+ in test_format_traceback_short",
            )

    def test_captured_traceback(self):
        # [FUNCTION DOC] Asserts a CapturedTraceback extracted here, when formatted, mentions
        # this test name.
        self.assertIn(
            "test_captured_traceback", "".join(CapturedTraceback.extract().format())
        )

    def test_captured_traceback_format_all(self):
        # [FUNCTION DOC] Formats two captured tracebacks together with format_all and asserts
        # two results are returned and the first mentions the format_all test name.
        rs = CapturedTraceback.format_all(
            [CapturedTraceback.extract(), CapturedTraceback.extract()]
        )
        self.assertEqual(len(rs), 2)
        self.assertIn("test_captured_traceback_format_all", "".join(rs[0]))

    def test_captured_traceback_format_all_cached(self):
        # [FUNCTION DOC] Same as the format_all test but formats one traceback beforehand so
        # format_all uses the cached formatting; asserts two results and the expected function
        # name in the first.
        tb = CapturedTraceback.extract()
        tb.format()  # cached
        rs = CapturedTraceback.format_all([tb, CapturedTraceback.extract()])
        self.assertEqual(len(rs), 2)
        self.assertIn("test_captured_traceback_format_all", "".join(rs[0]))


class TestTryImport(TestCase):
    # [CLASS DOC] Tests for torch._utils.try_import.
    def test_import_imported(self):
        # [FUNCTION DOC] Checks try_import returns the very same module object for a module
        # already imported (os).
        self.assertIn("os", sys.modules)
        os_module = try_import("os")
        self.assertIs(os_module, os)

    def test_import_existing(self):
        # [FUNCTION DOC] Checks try_import loads a not yet imported standard module (imaplib)
        # and returns it with real attributes.
        # KNOWN ISSUE: asserts imaplib is absent from the module table, so it fails if any
        # earlier code imported it.
        self.assertNotIn("imaplib", sys.modules)
        imaplib_module = try_import("imaplib")
        self.assertIsNotNone(imaplib_module)
        self.assertFalse(hasattr(imaplib_module, "not_attribute"))
        self.assertTrue(hasattr(imaplib_module, "IMAP4"))

    def test_import_missing(self):
        # [FUNCTION DOC] Checks try_import returns None for a nonexistent module name.
        missing_module = try_import("missing_module")
        self.assertIsNone(missing_module)


class TestUtilsInternal(TestCase):
    # [CLASS DOC] Tests for torch._utils_internal helpers.
    def test_max_clock_rate_falls_back_to_pynvml_when_nvidia_smi_missing(self):
        # [FUNCTION DOC] Fakes triton.testing, whose nvidia-smi query raises FileNotFoundError,
        # and a pynvml module recording calls, patches the module table, the hip version and the
        # pynvml handler getter, and asserts max_clock_rate returns the faked pynvml clock of
        # 1980, asks for the handler once, and the recorded calls are one max clock query with
        # the handle and the SM clock type followed by a shutdown. The cache of max_clock_rate
        # is cleared before and after.
        def nvsmi(_query):
            # [FUNCTION DOC] HELPER: Fake nvidia-smi query that always raises FileNotFoundError.
            raise FileNotFoundError("nvidia-smi")

        triton = types.ModuleType("triton")
        triton_testing = types.ModuleType("triton.testing")
        cast(Any, triton_testing).nvsmi = nvsmi
        cast(Any, triton).testing = triton_testing

        pynvml = types.ModuleType("pynvml")
        cast(Any, pynvml).NVML_CLOCK_SM = 1
        calls = []

        def nvmlDeviceGetMaxClockInfo(handle, clock_type):
            # [FUNCTION DOC] HELPER: Fake pynvml max clock query; records the handle and clock
            # type in the shared call list and returns 1980.
            calls.append(("max_clock", handle, clock_type))
            return 1980

        def nvmlShutdown():
            # [FUNCTION DOC] HELPER: Fake pynvml shutdown; records a shutdown entry in the
            # shared call list.
            calls.append("shutdown")

        cast(Any, pynvml).nvmlDeviceGetMaxClockInfo = nvmlDeviceGetMaxClockInfo
        cast(Any, pynvml).nvmlShutdown = nvmlShutdown

        torch._utils_internal.max_clock_rate.cache_clear()
        try:
            with (
                unittest.mock.patch.dict(
                    sys.modules,
                    {
                        "triton": triton,
                        "triton.testing": triton_testing,
                        "pynvml": pynvml,
                    },
                ),
                unittest.mock.patch.object(torch.version, "hip", None),
                unittest.mock.patch.object(
                    torch.cuda, "_get_pynvml_handler", return_value="handle"
                ) as get_pynvml_handler,
            ):
                self.assertEqual(torch._utils_internal.max_clock_rate(), 1980)
                get_pynvml_handler.assert_called_once_with()
        finally:
            torch._utils_internal.max_clock_rate.cache_clear()

        self.assertEqual(
            calls,
            [("max_clock", "handle", 1), "shutdown"],
        )


@deprecated()
def _deprecated_api(x, y=15):
    # [FUNCTION DOC] HELPER: Private function wrapped by the deprecated decorator, which at
    # import installs a public alias named without the underscore that warns about deprecation.
    # Returns the sum of its two arguments, the second defaulting to 15.
    return x + y


class TestDeprecate(TestCase):
    # [CLASS DOC] Tests for the deprecated decorator.
    def test_deprecated(self):
        # [FUNCTION DOC] Checks the public alias warns with a message saying it is deprecated
        # for positional and keyword calls, and that the private function can be called with no
        # warning check.
        with self.assertWarnsRegex(Warning, "is DEPRECATED"):
            # pyrefly: ignore [unknown-name]
            deprecated_api(1, 2)  # noqa: F821
        with self.assertWarnsRegex(Warning, "is DEPRECATED"):
            # pyrefly: ignore [unknown-name]
            deprecated_api(1, y=2)  # noqa: F821
        _deprecated_api(1, 2)
        _deprecated_api(1, y=2)


class TestDeviceLazyInit(TestCase):
    # [CLASS DOC] Test that device lazy initialization poisons forked subprocesses; instantiated
    # for non-CPU accelerators only.
    @unittest.skipIf(IS_WINDOWS, "pthread_atfork not available on Windows")
    def test_fork_poison_on_lazy_init(self, device):
        # [FUNCTION DOC] Initializes the device in the parent process, then in a forked child
        # tries to allocate on it and passes any exception back through a queue. Asserts the
        # parent received a RuntimeError whose message says the device cannot be re-initialized
        # in a forked subprocess and to use the spawn start method. Skipped on Windows.
        torch.empty(1, device=device)

        def child(q):
            # [FUNCTION DOC] HELPER: Child process body: allocates a tensor on the device and
            # puts any raised exception on the queue.
            try:
                torch.empty(1, device=device)
            except Exception as e:
                q.put(e)

        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        p.join()
        self.assertTrue(not q.empty())
        exc = q.get()
        pattern = (
            r"Cannot re-initialize .* in forked subprocess\. "
            r"To use .* with multiprocessing, you must use the 'spawn' start method"
        )
        self.assertIsInstance(exc, RuntimeError)
        self.assertRegex(str(exc), pattern)


instantiate_device_type_tests(
    TestDeviceLazyInit, globals(), except_for=["cpu"], allow_xpu=True
)


class TestEnv(TestCase):
    # [CLASS DOC] Tests for the environment variable helpers in torch._utils (getenv, setenv,
    # unsetenv and the os.environ hook).
    def test_getenv_matches_os(self):
        # [FUNCTION DOC] Checks getenv agrees with os.environ for PATH and for a name that does
        # not exist.
        for name in ("PATH", "PATH_DOES_NOT_EXIST_TORCH_TEST"):
            self.assertEqual(torch._utils.getenv(name), os.environ.get(name))

    def test_setenv_roundtrip(self):
        # [FUNCTION DOC] Sets a variable and reads it back through getenv.
        name = "TORCH_TEST_SETENV_VAR"
        torch._utils.setenv(name, "hello")
        self.assertEqual(torch._utils.getenv(name), "hello")

    def test_setenv_overwrite(self):
        # [FUNCTION DOC] Checks setenv with overwrite disabled keeps the old value, while the
        # default overwrites it.
        name = "TORCH_TEST_SETENV_OVERWRITE"
        torch._utils.setenv(name, "first")
        torch._utils.setenv(name, "second", overwrite=False)
        self.assertEqual(torch._utils.getenv(name), "first")
        torch._utils.setenv(name, "second")
        self.assertEqual(torch._utils.getenv(name), "second")

    def test_unsetenv(self):
        # [FUNCTION DOC] Sets a variable, then unsets it and checks getenv returns None.
        name = "TORCH_TEST_UNSETENV"
        torch._utils.setenv(name, "gone")
        self.assertEqual(torch._utils.getenv(name), "gone")
        torch._utils.unsetenv(name)
        self.assertIsNone(torch._utils.getenv(name))

    def test_os_environ_hook(self):
        # [FUNCTION DOC] Installs the os.environ hook, then checks writes and deletes via
        # os.environ are visible both in os.environ and through getenv, that installing again is
        # harmless, and after removal that os.putenv is no longer the torch hook.
        name = "TORCH_TEST_ENVIRON_HOOK"
        torch._utils.install_os_environ_hook()
        try:
            os.environ[name] = "hooked"
            # Both the cached os.environ dict and c10 observe the write.
            self.assertEqual(os.environ.get(name), "hooked")
            self.assertEqual(torch._utils.getenv(name), "hooked")
            del os.environ[name]
            self.assertIsNone(os.environ.get(name))
            self.assertIsNone(torch._utils.getenv(name))
            # Idempotent while installed.
            torch._utils.install_os_environ_hook()
        finally:
            torch._utils.remove_os_environ_hook()
        # After removal os.putenv is restored, so it is no longer our hook.
        self.assertIsNot(os.putenv, torch._utils._torch_putenv)


if __name__ == "__main__":
    run_tests()
