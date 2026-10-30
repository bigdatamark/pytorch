# Owner(s): ["module: nn"]
# [MODULE DOC] Device-generic tests that run over every ModuleInfo entry in module_db: forward
# and reference output, factory kwargs, device transfer, repr, pickling, inplace variants,
# non-contiguous inputs, gradcheck, CPU/accelerator parity, memory formats, train/eval flag
# detection, meta-device init, error inputs, to and to_empty. Also tests the JIT helper that
# replaces a submodule of a scripted module. Import-time effects: on ROCm builds sets the MIOpen
# NHWC environment variables, and the TestModule class is expanded per device through
# instantiate_device_type_tests (MPS and XPU allowed).

from itertools import chain, product
from inspect import signature, isgenerator
from copy import deepcopy
import tempfile
from operator import methodcaller

import torch

from torch._subclasses.meta_utils import assert_metadata_eq
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCPU, onlyAccelerator, toleranceOverride, tol, skipMeta, skipMPS)
from torch.testing._internal.common_modules import module_db, modules, ModuleErrorEnum, TrainEvalMode
from torch.testing._internal.common_utils import (
    TestCase, run_tests, freeze_rng_state, mock_wrapper, get_tensors_from, gradcheck,
    gradgradcheck, parametrize, wrapSwapTensorsTest, TEST_WITH_ROCM)
from unittest.mock import patch, call


if TEST_WITH_ROCM:
    import os
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM"] = "1"

class TestModule(TestCase):
    # [CLASS DOC] Parameterized module tests (decorated with the modules decorator, which
    # supplies device, dtype, module_info and training for each ModuleInfo, dtype and train or
    # eval mode). Class attributes enable the CUDA memory-leak check and non-default-stream runs
    # and set precision and relative tolerance to 1e-5. Helper methods handle recursive traversal
    # of nested tensor containers and gradient bookkeeping.
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    precision = 1e-5
    rel_tol = 1e-5

    def _assert_module_parameters_and_buffer_are(self, module, device, dtype):
        # [FUNCTION DOC] HELPER: Asserts every parameter and buffer of the module (recursive
        # named iteration) lives on the expected device (a string is converted to a device
        # object) and, for floating-point items only, has the expected dtype. Failures include
        # the item name.
        # Check device placement and dtype for created parameters and buffers.
        # Only verify floating point dtypes since that's what the kwarg or methods
        # such as `float()` applies to.
        if not isinstance(device, torch.device):
            device = torch.device(device)

        def _check_module(items, name, device=device, dtype=dtype):
            # [FUNCTION DOC] HELPER: Checks one iterable of name and tensor pairs against the
            # device and dtype captured from the enclosing scope; non-floating-point tensors skip
            # the dtype check.
            for item_name, item in items:
                self.assertEqual(
                    item.device, device,
                    lambda msg: f'{msg}\n{name} {item_name} is on device {item.device} instead of the expected device {device}')
                if item.dtype.is_floating_point:
                    self.assertEqual(
                        item.dtype, dtype,
                        lambda msg: f'{msg}\n{name} {item_name} is of dtype {item.dtype} instead of the expected dtype {dtype}')
        _check_module(module.named_parameters(), "Parameter")
        _check_module(module.named_buffers(), "Buffer")

    @modules(module_db)
    def test_forward(self, device, dtype, module_info, training):
        # [FUNCTION DOC] For each module input that has a forward input: under a frozen RNG
        # state, builds the module, moves it to device and dtype, sets train or eval mode and
        # runs forward. If the input supplies a reference function, the output must equal the
        # reference output. For float32 and float64 it also calls the matching conversion
        # method, reruns forward and checks all parameters and buffers have the expected device
        # and dtype.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        dtype_to_method_caller = {
            torch.float32: methodcaller("float"),
            torch.float64: methodcaller("double"),
        }
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)
                m.train(training)

                # === Do forward pass. ===
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                outputs = m(*args, **kwargs)

                # === Compare outputs to a reference if one is specified. ===
                # TODO: Handle precision
                reference_fn = module_input.reference_fn
                if reference_fn is not None:
                    ref_outputs = reference_fn(m, *args, **kwargs)
                    self.assertEqual(outputs, ref_outputs)

                # === Use the method call and verify the parameters and buffers ===
                if dtype in dtype_to_method_caller:
                    dtype_to_method_caller[dtype](m)
                    m(*args, **kwargs)
                    self._assert_module_parameters_and_buffer_are(m, device, dtype)

    # Tests passing factory kwargs (e.g. device / dtype) during module instantiation.
    # They should be applied to any created parameters and buffers.
    @modules(module_db)
    def test_factory_kwargs(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Skips module inputs whose construction creates no parameters or buffers
        # from new tensors (detected by wrapping Parameter creation and register_buffer with
        # pass-through mocks and comparing tensors in the call arguments with those passed to the
        # constructor). For the rest, adds device and dtype keyword arguments (mutating the
        # stored constructor kwargs) and rebuilds. Lazy modules must create all uninitialized
        # parameters and buffers with exactly those device and dtype keywords; others must end up
        # with every parameter and buffer on that device and dtype.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        for module_input in module_inputs:
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            # Check if this module creates parameters or registers buffers.
            # The mock magic here passes through to the real Parameter / register_buffer
            # logic and is only used to check call inputs.
            module_creates_params_or_buffers = False
            parameter_new = mock_wrapper(torch.nn.Parameter.__new__)
            with patch.object(torch.nn.Parameter, '__new__', parameter_new):
                register_buffer = mock_wrapper(torch.nn.Module.register_buffer)
                with patch.object(torch.nn.Module, 'register_buffer', register_buffer):
                    m = module_cls(*args, **kwargs)
                    m.train(training)

                    # Check if a parameter or buffer was created with a tensor not passed to the constructor.
                    constructor_tensors = get_tensors_from(args, kwargs)
                    for mock in [parameter_new.mock, register_buffer.mock]:
                        for call_args, call_kwargs in mock.call_args_list:
                            call_tensors = get_tensors_from(call_args, call_kwargs)
                            if len(call_tensors) > 0 and not constructor_tensors.intersection(call_tensors):
                                module_creates_params_or_buffers = True
                                break

            if not module_creates_params_or_buffers:
                continue

            # Instantiate module with the factory kwargs.
            kwargs.update({
                'device': device,
                'dtype': dtype,
            })

            if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
                # Ensure device and dtype are passed to all UninitializedParameters and UninitializedBuffers.
                uninit_param_new = mock_wrapper(torch.nn.UninitializedParameter.__new__)
                with patch.object(torch.nn.UninitializedParameter, '__new__', uninit_param_new):
                    uninit_buffer_new = mock_wrapper(torch.nn.UninitializedBuffer.__new__)
                    with patch.object(torch.nn.UninitializedBuffer, '__new__', uninit_buffer_new):
                        m = module_cls(*args, **kwargs)
                        m.train(training)
                        uninit_param_new.mock.assert_has_calls(
                            [call(device=device, dtype=dtype) for _ in uninit_param_new.mock.mock_calls])
                        uninit_buffer_new.mock.assert_has_calls(
                            [call(device=device, dtype=dtype) for _ in uninit_buffer_new.mock.mock_calls])
            else:
                # Check device placement and dtype for created parameters and buffers.
                # Only verify floating point dtypes since that's what the kwarg applies to.
                m = module_cls(*args, **kwargs)
                m.train(training)
                self._assert_module_parameters_and_buffer_are(m, device, dtype)

    @skipMPS
    @onlyAccelerator
    @modules(module_db)
    def test_multiple_device_transfer(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Accelerator only, not MPS. For inputs with a forward input, builds the
        # module on device, runs forward, moves it to CPU and runs with the matching CPU inputs,
        # moves it back and runs again, checking parameter and buffer placement after each step.
        # With two or more accelerators it also moves the module and inputs to device index 1,
        # runs forward under that device context and checks placement on that index.
        module_cls = module_info.module_cls
        module_inputs_device = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                              requires_grad=False, training=training)
        module_inputs_cpu = module_info.module_inputs_func(module_info, device="cpu", dtype=dtype,
                                                           requires_grad=False, training=training)
        for module_input_device, module_input_cpu in zip(module_inputs_device, module_inputs_cpu):
            if module_input_device.forward_input is None:
                continue

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input_device.constructor_input.args, module_input_device.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)
                m.train(training)

                # === Do forward pass on GPU ===
                input_device_args = module_input_device.forward_input.args
                input_device_kwargs = module_input_device.forward_input.kwargs
                m(*input_device_args, **input_device_kwargs)
                self._assert_module_parameters_and_buffer_are(m, device, dtype)

                # === Move to CPU ===
                input_cpu_args = module_input_cpu.forward_input.args
                input_cpu_kwargs = module_input_cpu.forward_input.kwargs
                m.cpu()
                m(*input_cpu_args, **input_cpu_kwargs)
                self._assert_module_parameters_and_buffer_are(m, "cpu", dtype)

                # === Move back to GPU and forward pass ===
                m.to(device)
                m(*input_device_args, **input_device_kwargs)
                self._assert_module_parameters_and_buffer_are(m, device, dtype)

                if torch.accelerator.device_count() >= 2:
                    # === test cross-GPU transfer works
                    def _to_device1(objs):
                        # [FUNCTION DOC] HELPER: Recursively moves tensors inside tuples, lists
                        # and dicts to device index 1, rebuilding containers with their original
                        # type; other objects are returned as is.
                        if isinstance(objs, (tuple, list)):
                            return type(objs)(_to_device1(item) for item in objs)
                        elif isinstance(objs, dict):
                            return {name: _to_device1(item) for name, item in objs.items()}
                        elif isinstance(objs, torch.Tensor):
                            return objs.to(1)
                        else:
                            return objs
                    input_device_1_args = _to_device1(input_device_args)
                    input_device_1_kwargs = _to_device1(input_device_kwargs)

                    m.to(1)
                    with torch.accelerator.device_index(1):
                        m(*input_device_1_args, **input_device_1_kwargs)
                    self._assert_module_parameters_and_buffer_are(m, torch.device(f"{torch.device(device).type}:1"), dtype)

    @modules(module_db)
    def test_repr(self, device, dtype, module_info, training):
        # [FUNCTION DOC] For every module input, constructs the module, moves it to device and
        # dtype, sets train or eval mode and calls the repr and str conversions. Passes if
        # neither raises.
        # Test module can be represented with repr and str without errors.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        for module_input in module_inputs:
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            m.train(training)

            # Check that these methods do not raise errors
            m.__repr__()
            str(m)

    @modules(module_db)
    def test_save_load(self, device, dtype, module_info, training):
        # [FUNCTION DOC] For inputs with a forward input: builds the module under a frozen RNG
        # state, takes its state dict, runs forward, then saves the whole module to a temporary
        # file, loads it back with weights_only disabled and checks the copy gives an equal
        # output. Also saves the state dict and checks it loads equal both with the default
        # loader and with weights_only enabled.
        # Test that module can be pickled and unpickled.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)
                m.train(training)
                sd = m.state_dict()

                # === Do forward pass. ===
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                output = m(*args, **kwargs)

                # === Check saved/loaded module gives the same output. ===
                with tempfile.TemporaryFile() as f:
                    torch.save(m, f)
                    f.seek(0)
                    # weights_only=False as this is legacy code that saves the model
                    m_copy = torch.load(f, weights_only=False)
                    output_from_copy = m_copy(*args, **kwargs)
                    self.assertEqual(output, output_from_copy)

                # === Check saved/loaded state_dict are the same (including weights_only load). ===
                with tempfile.TemporaryFile() as f:
                    torch.save(sd, f)
                    f.seek(0)
                    sd_copy = torch.load(f)
                    self.assertEqual(sd_copy, sd)
                    del sd_copy
                    f.seek(0)
                    sd_copy_wo = torch.load(f, weights_only=True)
                    self.assertEqual(sd_copy_wo, sd)

    @skipMeta
    @modules([module_info for module_info in module_db
              if 'inplace' in signature(module_info.module_cls).parameters])
    def test_check_inplace(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Skipped for meta. Runs only for modules whose constructor accepts an
        # inplace argument. Builds out-of-place and inplace variants and checks that the
        # out-of-place run leaves the first input version counter unchanged, the inplace run on a
        # cloned copy bumps it, the outputs are equal, and after backward with the same random
        # gradient the input gradients agree.
        # KNOWN ISSUE: first_param_name is a name and parameter pair taken from the signature
        # items, so the check that it is absent from the input keyword arguments can never fail.
        # Check if the inplace variant of the module gives the same result as the out of place
        # variant.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m_op = module_cls(*args, **kwargs, inplace=False)
            m_op.to(device).to(dtype)
            m_op.train(training)
            m_inplace = module_cls(*args, **kwargs, inplace=True)
            m_inplace.to(device).to(dtype)
            m_inplace.train(training)

            # === Inplace modules only supports inplace operations on the first argument ===
            input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs

            # ===  Do not allow the first input to be in input_kwargs ===
            forward_sig = signature(m_op).parameters
            self.assertGreaterEqual(len(forward_sig), 1)
            first_param_name = next(iter(forward_sig.items()))
            self.assertNotIn(first_param_name, input_kwargs)

            # === Out of place operation does not write to original tensor ===
            self.assertGreaterEqual(len(input_args), 1)
            input_version = input_args[0]._version
            with freeze_rng_state():
                output_op = m_op(*input_args, **input_kwargs)
            self.assertEqual(input_args[0]._version, input_version)

            # === Check that the inplace operation gives the same result ===
            input_arg_copy = deepcopy(input_args)
            input_arg_clone = tuple(i.clone() for i in input_arg_copy)
            input_clone_version = input_arg_clone[0]._version
            with freeze_rng_state():
                output_ip = m_inplace(*input_arg_clone, **input_kwargs)
            self.assertGreater(input_arg_clone[0]._version, input_clone_version)
            self.assertEqual(output_op, output_ip)

            # === Check that the gradients are the same ===
            grad = output_op.data.clone().normal_()
            output_op.backward(grad)
            output_ip.backward(grad)
            self.assertEqual(input_args[0].grad, input_arg_copy[0].grad)

    def _traverse_obj(self, obj, func):
        # [FUNCTION DOC] HELPER: Recursively applies a function to every tensor or Parameter
        # found in tuples and lists (type preserved), generators (consumed into a tuple) and
        # dict values (dict rebuilt). Returns the rebuilt structure; other objects are passed
        # through unchanged.
        if isinstance(obj, (tuple, list)):
            return type(obj)(self._traverse_obj(o, func) for o in obj)
        elif isgenerator(obj):
            return tuple(self._traverse_obj(o, func) for o in obj)
        elif isinstance(obj, dict):
            return {name: self._traverse_obj(o, func) for name, o in obj.items()}
        elif isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
            return func(obj)
        else:
            return obj

    def _retain_grad(self, obj):
        # [FUNCTION DOC] HELPER: Calls retain_grad on every tensor in the nested structure that
        # requires grad so non-leaf gradients are populated after backward. Mutates tensors only.
        # gradients needs to be retained to check for grad. This is useful when
        # non-leaves are present in the graph.
        def inner_retain_grad(obj):
            # [FUNCTION DOC] HELPER: Calls retain_grad on the tensor if it requires grad; no
            # return value.
            if obj.requires_grad:
                obj.retain_grad()
        self._traverse_obj(obj, inner_retain_grad)

    def _get_grads(self, obj):
        # [FUNCTION DOC] HELPER: Returns a structure mirroring the input holding the grad of each
        # tensor that requires grad (None for tensors that do not).
        def inner_get_grad(obj):
            # [FUNCTION DOC] HELPER: Returns the grad of the tensor if it requires grad,
            # else None.
            if obj.requires_grad:
                return obj.grad
        return self._traverse_obj(obj, inner_get_grad)

    def _zero_grad(self, obj):
        # [FUNCTION DOC] HELPER: Resets the grad attribute to None on every tensor in the nested
        # structure (generators are consumed). Mutates tensors; returns nothing.
        def inner_zero_grad(obj):
            # [FUNCTION DOC] HELPER: Sets the tensor grad to None if one is present.
            if obj.grad is not None:
                obj.grad = None
        self._traverse_obj(obj, inner_zero_grad)

    @modules(module_db)
    def test_non_contiguous_tensors(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Checks module results do not depend on input or gradient strides. For
        # each input with at least one non-scalar tensor, computes the default output, a random
        # output gradient, input gradients and parameter gradients. Then, for every combination
        # of contiguous or non-contiguous inputs and contiguous or non-contiguous output
        # gradients, it reruns forward and backward and requires equal outputs, input gradients
        # (absolute tolerance 1e-4) and parameter gradients.
        # Check modules work with non-contiguous tensors

        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)

        def _make_non_contiguous(obj):
            # [FUNCTION DOC] HELPER: Returns a copy of the nested structure in which every
            # non-scalar tensor is replaced by a detached, non-contiguous tensor with the same
            # values and the same requires_grad flag.
            def inner_make_non_contiguous(obj):
                # [FUNCTION DOC] HELPER: Leaves non-tensors and zero-dimensional tensors
                # untouched. Otherwise duplicates each element along the last dimension and takes
                # every other element by slicing, yielding equal values in a strided layout;
                # detaches and restores requires_grad.
                # Scalar tensors can not be made non-contiguous
                if not isinstance(obj, torch.Tensor) or obj.dim() == 0:
                    return obj

                out = torch.repeat_interleave(obj, 2, dim=-1)
                out = out[..., ::2].detach()
                out.requires_grad = obj.requires_grad
                return out
            return self._traverse_obj(obj, inner_make_non_contiguous)

        def _can_be_noncontiguous(obj):
            # [FUNCTION DOC] HELPER: Returns True if the nested tuple, list or dict structure
            # contains at least one tensor with one or more dimensions, else False.
            if isinstance(obj, (tuple, list)):
                return any(_can_be_noncontiguous(o) for o in obj)
            elif isinstance(obj, dict):
                return any(_can_be_noncontiguous(o) for o in obj.values())
            # scalar tensors can not be non-contiguous
            return isinstance(obj, torch.Tensor) and obj.dim() != 0

        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
            if not (_can_be_noncontiguous(input_args) or _can_be_noncontiguous(input_kwargs)):
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            m.train(training)

            self._retain_grad((input_args, input_kwargs))

            # === Forward with default input
            with freeze_rng_state():
                default_output = m(*input_args, **input_kwargs)
                if isinstance(default_output, torch.Tensor):
                    grad_output = default_output.clone().detach_().normal_()
                    default_output.backward(grad_output, retain_graph=True)
                else:
                    grad_output = tuple(self._traverse_obj(o, lambda o: o.clone().detach_().normal_() if o.requires_grad else None)
                                        for o in default_output)
                    flattened_default_output = torch.utils._pytree.tree_leaves(default_output)
                    flattened_grad_output = torch.utils._pytree.tree_leaves(grad_output)
                    for o, g_o in zip(flattened_default_output, flattened_grad_output):
                        if (o.requires_grad):
                            o.backward(g_o, retain_graph=True)

            default_input_args_grad, default_input_kwargs_grad = deepcopy(self._get_grads((input_args, input_kwargs)))
            default_param_grad = deepcopy([p.grad for p in m.parameters()])

            # === Construct non-contiguous tensors ===
            nc_input_args, nc_input_kwargs = _make_non_contiguous((input_args, input_kwargs))
            nc_grad_output = _make_non_contiguous(grad_output)

            # === Compare results with non-contiguous and contiguous tensors ===
            inputs = [(input_args, input_kwargs), (nc_input_args, nc_input_kwargs)]
            grads = [grad_output, nc_grad_output]

            for (in_args, in_kwargs), g_out in product(inputs, grads):
                g_out_copy = deepcopy(g_out)
                self._zero_grad((in_args, in_kwargs))
                self._zero_grad(m.parameters())

                with freeze_rng_state():
                    out = m(*in_args, **in_kwargs)
                    if isinstance(out, torch.Tensor):
                        out.backward(g_out_copy, retain_graph=True)
                    else:
                        flattened_out = torch.utils._pytree.tree_leaves(out)
                        flattened_g_out_copy = torch.utils._pytree.tree_leaves(g_out_copy)
                        for o, g_o in zip(flattened_out, flattened_g_out_copy):
                            if o.requires_grad:
                                o.backward(g_o, retain_graph=True)

                input_args_grad, input_kwargs_grad = self._get_grads((in_args, in_kwargs))
                self.assertEqual(out, default_output)
                self.assertEqual(input_args_grad, default_input_args_grad, atol=1e-4, rtol=0)
                self.assertEqual(input_kwargs_grad, default_input_kwargs_grad, atol=1e-4, rtol=0)

                param_grad = [p.grad for p in m.parameters()]
                self.assertEqual(param_grad, default_param_grad)

    def _test_gradients_helper(self, device, dtype, module_info, training, check):
        # [FUNCTION DOC] HELPER: Shared body of the gradient tests. Skips XPU MultiheadAttention.
        # Uses the module-specific nondeterminism tolerance only on CUDA with cuDNN enabled or on
        # XPU, otherwise zero. For each input with a forward input, builds the module (lazy
        # modules get one no-grad forward to initialize), then runs the given check function
        # (gradcheck or gradgradcheck) over all inputs, parameters and tensor keyword arguments
        # together, and then with each parameter and tensor keyword argument enabled one at a
        # time. Asserts every check passes. Parameters and tensor keyword arguments are left not
        # requiring grad at the end.
        # Check gradients
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)
        if "xpu" in device and module_info.name == "nn.MultiheadAttention":
            self.skipTest("GradcheckError issue in MultiheadAttention, https://github.com/intel/torch-xpu-ops/issues/2356")
        # === Set nondet tol for gradcheck to user-defined value if on CUDA and cudNN is enabled
        gradcheck_nondet_tol = 0.0
        if (torch.device(device).type == 'cuda' and torch.backends.cudnn.enabled) or torch.device(device).type == 'xpu':
            gradcheck_nondet_tol = module_info.gradcheck_nondet_tol

        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            m.train(training)

            params = tuple(m.parameters())

            # === Lazy modules need to see an input to initialize params before gradcheck is run. ===
            input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
            if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
                with torch.no_grad():
                    m(*input_args, **input_kwargs)

            # === Perform gradient check on the input_args ===
            other_kwargs = {}
            kwarg_tensors = []
            for name, obj in input_kwargs.items():
                if isinstance(obj, torch.Tensor):
                    kwarg_tensors.append((name, obj))
                else:
                    other_kwargs[name] = obj

            def fn_to_gradcheck(*flat_input_and_params):
                # [FUNCTION DOC] HELPER: Function under gradient check. Rebuilds the structured
                # inputs from the flat tensor list using a spec computed later in the enclosing
                # scope, splits them into positional inputs and tensor keyword arguments, runs
                # the module forward under a frozen RNG state with the non-tensor keyword
                # arguments, and returns the flattened output tensors.
                input_and_params = torch.utils._pytree.tree_unflatten(flat_input_and_params, flat_spec)
                new_input_args = input_and_params[:len(input_args)]
                kwarg_args = input_and_params[-len(kwarg_tensors):]
                new_kwargs = {name: obj for (name, _), obj in zip(kwarg_tensors, kwarg_args)}

                with freeze_rng_state():
                    output = m(*new_input_args, **new_kwargs, **other_kwargs)
                    output_flattened = torch.utils._pytree.tree_leaves(output)
                    return output_flattened

            def do_check(flat_input):
                # [FUNCTION DOC] HELPER: Asserts the chosen check function returns true for the
                # flat input list, passing the nondeterminism tolerance and the fast mode flag
                # from the ModuleInfo.
                self.assertTrue(
                    check(
                        fn_to_gradcheck,
                        flat_input,
                        nondet_tol=gradcheck_nondet_tol,
                        fast_mode=module_info.gradcheck_fast_mode
                    ))

            # check total derivative
            grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
            flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
            do_check(flat_input)

            # check partial derivatives
            old_params_requires_grad = [p.requires_grad for p in params]
            for p in params:
                p.requires_grad = False

            old_kwargs_requires_grad = [obj.requires_grad for (_, obj) in kwarg_tensors]
            for (_, obj) in kwarg_tensors:
                obj.requires_grad = False

            for p, old in zip(params, old_params_requires_grad):
                p.requires_grad = old
                grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
                flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
                do_check(flat_input)
                p.requires_grad = False

            for (_, obj), old in zip(kwarg_tensors, old_kwargs_requires_grad):
                obj.requires_grad = old
                grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
                flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
                do_check(flat_input)
                obj.requires_grad = False

    @modules(module_db, allowed_dtypes=[torch.double])
    def test_grad(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Runs the gradient helper with gradcheck for double dtype only.
        self._test_gradients_helper(device, dtype, module_info, training, gradcheck)

    @modules([m for m in module_db if m.supports_gradgrad],
             allowed_dtypes=[torch.double])
    def test_gradgrad(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Runs the gradient helper with gradgradcheck for double dtype only,
        # limited to ModuleInfo entries that support double backward.
        self._test_gradients_helper(device, dtype, module_info, training, gradgradcheck)

    @skipMPS
    @onlyAccelerator
    @with_tf32_off  # Turn off TF32 to compute at full precision https://github.com/pytorch/pytorch/issues/86798
    @toleranceOverride({torch.float32: tol(5e-2, 0),
                        torch.float64: tol(4e-4, 0)})
    @modules(module_db)
    def test_cpu_gpu_parity(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Accelerator only, not MPS; TF32 disabled and per-dtype tolerance
        # overrides applied. Returns early for RNN, GRU and LSTM in eval mode on CUDA with cuDNN.
        # Builds the CPU and device modules from the same constructor arguments, copies CPU
        # parameter data to the device module, and checks forward outputs are equal. Then, five
        # times, backpropagates a random gradient through both and compares input, parameter and
        # keyword input gradients.
        # KNOWN ISSUE: the forward input is used without a check for None, unlike sibling tests.
        # TODO: RNN / GRU / LSTM don't support backwards on eval mode for cuDNN; skip this in a
        # nicer way for eval mode only.
        # See https://github.com/pytorch/pytorch/issues/79161
        rnn_modules = {torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM}
        if (module_info.module_cls in rnn_modules
                and not training
                and 'cuda' in device
                and torch.backends.cudnn.enabled):
            return

        # Test cpu and gpu results are the same
        module_cls = module_info.module_cls
        module_inputs_cpu = module_info.module_inputs_func(module_info, device="cpu", dtype=dtype,
                                                           requires_grad=True, training=training)

        def _to_device(obj):
            # [FUNCTION DOC] HELPER: Copies a nested structure to the target device. Tensors are
            # detached, moved and given the original requires_grad flag; tuples and dicts are
            # handled recursively; any other object (including lists) is deep-copied without
            # being moved.
            if isinstance(obj, torch.Tensor):
                res = obj.detach().to(device=device)
                res.requires_grad = obj.requires_grad
                return res
            elif isinstance(obj, tuple):
                return tuple(_to_device(o) for o in obj)
            elif isinstance(obj, dict):
                return {key: _to_device(o) for key, o in obj.items()}
            else:
                return deepcopy(obj)

        for module_input in module_inputs_cpu:
            # === Move input from cpu to device ===
            cpu_forward_args = module_input.forward_input.args
            cpu_forward_kwargs = module_input.forward_input.kwargs

            gpu_forward_args, gpu_forward_kwargs = _to_device((cpu_forward_args, cpu_forward_kwargs))

            self._retain_grad((cpu_forward_args, cpu_forward_kwargs, gpu_forward_args, gpu_forward_kwargs))

            # === Construct module on cpu and gpu ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            cpu_module = module_cls(*args, **kwargs).to(dtype).to("cpu")
            cpu_module.train(training)
            gpu_module = module_cls(*args, **kwargs).to(dtype).to(device)
            gpu_module.train(training)

            # === Lazy modules need to see an input to initialize params ===
            if issubclass(module_cls, torch.nn.modules.lazy.LazyModuleMixin):
                with torch.no_grad():
                    cpu_module(*cpu_forward_args, **cpu_forward_kwargs)
                    gpu_module(*gpu_forward_args, **gpu_forward_kwargs)

            for cpu_p, gpu_p in zip(cpu_module.parameters(), gpu_module.parameters()):
                gpu_p.data.copy_(cpu_p)

            # === Compare forward output between cpu and gpu ===
            cpu_outputs = cpu_module(*cpu_forward_args, **cpu_forward_kwargs)
            gpu_outputs = gpu_module(*gpu_forward_args, **gpu_forward_kwargs)

            self.assertEqual(cpu_outputs, gpu_outputs)

            # === Run backwards on CPU and GPU and compare results ===
            def check_backward(cpu_output, gpu_output):
                # [FUNCTION DOC] HELPER: Backpropagates one random gradient (generated on the CPU
                # output and converted to the device output type) through both outputs, retaining
                # the graph, then asserts equal gradients for positional inputs, every parameter
                # and keyword inputs.
                cpu_grad_output = cpu_output.clone().normal_()
                gpu_grad_output = cpu_grad_output.type_as(gpu_output)

                cpu_output.backward(cpu_grad_output, retain_graph=True)
                gpu_output.backward(gpu_grad_output, retain_graph=True)

                cpu_grad_input = self._get_grads(cpu_forward_args)
                gpu_grad_input = self._get_grads(gpu_forward_args)
                self.assertEqual(cpu_grad_input, gpu_grad_input)

                for cpu_p, gpu_p in zip(cpu_module.parameters(), gpu_module.parameters()):
                    self.assertEqual(cpu_p.grad, gpu_p.grad)

                cpu_grad_kwarg_input = self._get_grads(cpu_forward_kwargs)
                gpu_grad_kwarg_input = self._get_grads(gpu_forward_kwargs)
                self.assertEqual(cpu_grad_kwarg_input, gpu_grad_kwarg_input)

            for _ in range(5):
                if isinstance(cpu_outputs, torch.Tensor):
                    check_backward(cpu_outputs, gpu_outputs)
                else:
                    flatten_cpu_outputs = torch.utils._pytree.tree_leaves(cpu_outputs)
                    flatten_gpu_outputs = torch.utils._pytree.tree_leaves(gpu_outputs)
                    for cpu_output, gpu_output in zip(flatten_cpu_outputs, flatten_gpu_outputs):
                        if cpu_output.requires_grad:
                            check_backward(cpu_output, gpu_output)

    @with_tf32_off
    @modules(module_db)
    def test_memory_format(self, device, dtype, module_info, training):
        # [FUNCTION DOC] TF32 disabled. Uses looser tolerances on CUDA devices with compute
        # capability 8.0 or 8.6. For each input computes a reference output in the all-contiguous
        # configuration and, in training mode, reference gradients from random output gradients.
        # Then loops over input memory formats and module memory formats (channels_last variants
        # are chosen by whether an input has 4 or 5 dimensions), converts inputs and module, and
        # checks outputs and gradients equal the reference (unless both formats are contiguous)
        # and that output and gradient memory formats match the expectation. Overwrites the
        # stored forward inputs with the converted tensors as it goes.
        is_sm86or80 = torch.device(device).type == "cuda" and (torch.cuda.get_device_capability(0) == (8, 6)
                                                               or torch.cuda.get_device_capability(0) == (8, 0))
        # TODO tighten it to a specific module
        atol, rtol = (3e-3, 7e-3) if is_sm86or80 else (None, None)
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)
        module_memformat_affects_out = module_info.module_memformat_affects_out

        def _get_mem_formats(channels_last=False, channels_last_3d=False):
            # [FUNCTION DOC] HELPER: Returns a pair of lists: input memory formats to try and
            # module memory formats to try. The channels_last (4D) and channels_last_3d (5D)
            # variants add that format; otherwise only contiguous inputs, and preserve or
            # contiguous modules.
            if channels_last:
                return ([torch.contiguous_format, torch.channels_last],
                        [torch.preserve_format, torch.contiguous_format, torch.channels_last])
            elif channels_last_3d:
                return ([torch.contiguous_format, torch.channels_last_3d],
                        [torch.preserve_format, torch.contiguous_format, torch.channels_last_3d])
            else:
                return ([torch.contiguous_format],
                        [torch.preserve_format, torch.contiguous_format])

        # Check that at least one Tensor input has dim == n
        def _check_dims(obj, n):
            # [FUNCTION DOC] HELPER: Returns True if any tensor in the nested tuple or list has
            # exactly the given number of dimensions; dicts and other objects count as no match.
            if isinstance(obj, torch.Tensor):
                return obj.dim() == n
            elif isinstance(obj, (tuple, list)):
                return any(_check_dims(o, n) for o in obj)
            else:
                return False

        # Called after _check_dims, when we know that >= 1 tensor can be converted to mem_format
        def _to_mem_format(mem_format, obj):
            # [FUNCTION DOC] HELPER: Returns a copy of the nested structure with every
            # tensor converted to the requested memory format as a detached tensor
            # keeping requires_grad.
            def inner_to_mem_format(obj):
                # [FUNCTION DOC] HELPER: Converts one tensor to the target format; if the format
                # is channels_last without 4 dimensions or channels_last_3d without 5, only
                # clones and detaches.
                d = obj.dim()
                if ((mem_format == torch.channels_last and d != 4)
                   or (mem_format == torch.channels_last_3d and d != 5)):
                    return obj.detach().clone().requires_grad_(obj.requires_grad)
                return obj.clone().to(memory_format=mem_format).detach().requires_grad_(obj.requires_grad)

            return self._traverse_obj(obj, inner_to_mem_format)

        def _check_out_mem_format(output, input_mem_format, module_mem_format):
            # [FUNCTION DOC] HELPER: Asserts the memory format of each tensor in the nested
            # output. Expected channels_last for 4D (or channels_last_3d for 5D) when the input
            # has that format, or when the module has it and the ModuleInfo says it affects the
            # output; empty tensors always pass; everything else must simply be contiguous.
            def inner_check_out_mem_format(output):
                # [FUNCTION DOC] HELPER: Performs the per-tensor memory format assertion
                # described by the enclosing function, based on the tensor dimension count.
                d = output.dim()
                if (d == 4 and ((input_mem_format == torch.channels_last)
                                or (module_mem_format == torch.channels_last and module_memformat_affects_out))):
                    self.assertTrue(output.numel() == 0 or output.is_contiguous(memory_format=torch.channels_last))
                elif (d == 5 and ((input_mem_format == torch.channels_last_3d)
                                  or (module_mem_format == torch.channels_last_3d and module_memformat_affects_out))):
                    self.assertTrue(output.numel() == 0 or output.is_contiguous(memory_format=torch.channels_last_3d))
                else:
                    self.assertTrue(output.is_contiguous())
            return self._traverse_obj(output, inner_check_out_mem_format)

        def _req_grad(t):
            # [FUNCTION DOC] HELPER: Returns True if the object is a tensor that requires grad.
            return isinstance(t, torch.Tensor) and t.requires_grad

        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            supports_channels_last = _check_dims(module_input.forward_input.args, 4)
            supports_channels_last_3d = _check_dims(module_input.forward_input.args, 5)
            input_mem_formats, module_mem_formats = _get_mem_formats(supports_channels_last, supports_channels_last_3d)

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)
                m.train(training)

                # === Get output in (contiguous, contiguous) configuration. ===
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                desired_outputs = m(*args, **kwargs)
                # === Do backward pass. ===
                ref_diff_outputs = tuple(t for t in torch.utils._pytree.tree_leaves(desired_outputs) if _req_grad(t))
                if training and len(ref_diff_outputs) > 0:
                    params = tuple(p for p in m.parameters())
                    ref_diff_inputs = tuple(
                        t
                        for t in torch.utils._pytree.tree_leaves((args, kwargs, params))
                        if _req_grad(t)
                    )
                    ref_grad_outputs = tuple(
                        torch.rand_like(t)
                        for t in ref_diff_outputs
                    )
                    ref_grad_inputs = torch.autograd.grad(
                        ref_diff_outputs,
                        ref_diff_inputs,
                        grad_outputs=ref_grad_outputs,
                    )

                for input_mem_format in input_mem_formats:
                    # === Change memformat of input. ===
                    d_args = _to_mem_format(input_mem_format, module_input.forward_input.args)
                    d_kwargs = _to_mem_format(input_mem_format, module_input.forward_input.kwargs)

                    # See https://github.com/pytorch/pytorch/issues/107861
                    # When inductor tests are turned on, the setting of requires_grad will be lost
                    for t1, t2 in zip(
                        torch.utils._pytree.tree_leaves(d_args),
                        torch.utils._pytree.tree_leaves(module_input.forward_input.args),
                    ):
                        t1.requires_grad_(t2.requires_grad)
                    for t1, t2 in zip(
                        torch.utils._pytree.tree_leaves(d_kwargs),
                        torch.utils._pytree.tree_leaves(module_input.forward_input.kwargs),
                    ):
                        t1.requires_grad_(t2.requires_grad)

                    module_input.forward_input.args = d_args
                    module_input.forward_input.kwargs = d_kwargs

                    for module_mem_format in module_mem_formats:
                        # === Change memformat of module ===
                        m.to(memory_format=module_mem_format)

                        # === Do forward pass. ===
                        args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                        outputs = m(*args, **kwargs)

                        # === Compare outputs to (contiguous, contiguous) output. ===
                        if input_mem_format != torch.contiguous_format or module_mem_format != torch.contiguous_format:
                            self.assertEqual(outputs, desired_outputs, rtol=rtol, atol=atol)

                        # === Check mem format of output. ===
                        _check_out_mem_format(outputs, input_mem_format, module_mem_format)

                        # === Do backward pass. ===
                        diff_outputs = tuple(t for t in torch.utils._pytree.tree_leaves(outputs) if _req_grad(t))
                        if training and len(diff_outputs) > 0:
                            params = tuple(p for p in m.parameters())
                            diff_inputs = tuple(
                                t
                                for t in torch.utils._pytree.tree_leaves((args, kwargs, params))
                                if _req_grad(t)
                            )
                            grad_outputs = tuple(
                                torch.empty_like(t1).copy_(t2)
                                for (t1, t2) in zip(diff_outputs, ref_grad_outputs)
                            )

                            grad_inputs = torch.autograd.grad(
                                diff_outputs,
                                diff_inputs,
                                grad_outputs=grad_outputs,
                            )

                            if (
                                input_mem_format != torch.contiguous_format
                                or module_mem_format != torch.contiguous_format
                            ):
                                self.assertEqual(
                                    grad_inputs, ref_grad_inputs, rtol=rtol, atol=atol
                                )

                            # === Check mem format of grad_inputs. ===
                            _check_out_mem_format(grad_inputs, input_mem_format, module_mem_format)

    # Test whether train and eval modes differ for each module. Use to verify
    # that the ModuleInfo entry flag is correct.
    @modules(module_db, train_eval_mode=TrainEvalMode.train_only)
    def test_if_train_and_eval_modes_differ(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Train-only mode. For each input with a forward input, builds the module,
        # deletes its training attribute and runs forward. If an AttributeError mentions
        # training, asserts the ModuleInfo train_and_eval_differ flag is true (with an
        # explanatory message); any other AttributeError is re-raised. Nothing is asserted when
        # forward succeeds.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)

        # Run forward inputs through to see if the training flag is accessed during forward.
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            m.train(training)

            # Remove training attribute and see if forward still works.
            delattr(m, 'training')

            # === Do forward pass. ===
            try:
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                m(*args, **kwargs)
            except AttributeError as e:
                if "'training'" in str(e):
                    self.assertTrue(module_info.train_and_eval_differ,
                                    lambda msg: f"{msg}\nThe ModuleInfo entry for {module_info.name} has "
                                    "train_and_eval_differ=False, but the training mode was found to "
                                    "affect the forward pass. Consider setting train_and_eval_differ=True "
                                    "for this ModuleInfo entry.")
                else:
                    raise e


    @onlyCPU
    @modules(module_db)
    def test_device_ctx_init(self, device, dtype, module_info, training):
        # [FUNCTION DOC] CPU only. Builds each module twice from the same inputs: normally on CPU
        # and inside a meta device context with inputs generated under that context. For every
        # parameter and buffer pair (lazy ones skipped) asserts the meta tensor is on the meta
        # device and its metadata matches the CPU tensor.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        with torch.device('meta'):
            module_inputs_meta = module_info.module_inputs_func(module_info, device=None, dtype=dtype,
                                                                requires_grad=False, training=training)

        for module_input, module_input_meta in zip(module_inputs, module_inputs_meta):
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            c_args_meta, c_kwargs_meta = module_input_meta.constructor_input.args, module_input_meta.constructor_input.kwargs

            m_cpu = module_cls(*c_args, **c_kwargs)

            with torch.device('meta'):
                m = module_cls(*c_args_meta, **c_kwargs_meta)

            for (p_meta, p_cpu) in chain(zip(m.parameters(), m_cpu.parameters()),
                                         zip(m.buffers(), m_cpu.buffers())):
                if torch.nn.parameter.is_lazy(p_meta):
                    continue
                self.assertTrue(p_meta.is_meta)
                assert_metadata_eq(self.assertEqual, p_meta, p_cpu)


    @modules([module for module in module_db if module.module_error_inputs_func is not None])
    def test_errors(self, device, dtype, module_info, training):
        # [FUNCTION DOC] Runs only for ModuleInfo entries with an error-inputs function. For each
        # error input, construction errors must raise the given exception matching the regex
        # during construction; forward errors build the module first and must raise during
        # forward. Any other error kind raises NotImplementedError.
        module_cls = module_info.module_cls
        error_inputs = module_info.module_error_inputs_func(module_info, device=device, dtype=dtype,
                                                            requires_grad=False, training=training)
        for error_input in error_inputs:
            module_input = error_input.module_error_input
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            if error_input.error_on == ModuleErrorEnum.CONSTRUCTION_ERROR:
                with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                    m = module_cls(*c_args, **c_kwargs)
            elif error_input.error_on == ModuleErrorEnum.FORWARD_ERROR:
                m = module_cls(*c_args, **c_kwargs)
                fw_args, fw_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                    m(*fw_args, **fw_kwargs)
            else:
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")

    # Only run this test for float32 because the test loops over all the dtypes
    @skipMPS
    @modules([module for module in module_db if not module.is_lazy], allowed_dtypes=[torch.float32])
    @parametrize('swap', [True, False])
    @parametrize('set_grad', [True, False])
    @wrapSwapTensorsTest()
    def test_to(self, device, dtype, module_info, training, swap, set_grad):
        # [FUNCTION DOC] Float32 only, not MPS, run for swap on or off and set_grad on or off.
        # Sets the global swap-on-conversion flag. Builds each module, replaces its parameters
        # and buffers by copies on the device and dtype (optionally with random grads),
        # optionally runs forward and backward then converts to half and back (swap, no
        # grads). Then for every device and dtype combination converts the module and asserts
        # parameter type, device and dtype, and identity: with swap the Python ids stay the
        # same while the underlying tensor handles change; without swap both stay the same,
        # and gradients likewise.
        # KNOWN ISSUE: forward input is read without a check for None for each module input.
        module_cls = module_info.module_cls
        devices = ['cpu']
        if torch.accelerator.is_available():
            devices.append(torch.device(device).type)
        dtypes = module_info.dtypes
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        torch.__future__.set_swap_module_params_on_conversion(swap)

        for module_input in module_inputs:
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs

            m = module_cls(*c_args, **c_kwargs)

            # Avoid using `module.to()` when constructing module since that is the method we are testing
            def _to(m, set_grad=False):
                # [FUNCTION DOC] HELPER: Recursively (children first) replaces the direct
                # parameters of a module with new Parameters detached, cloned and moved to the
                # device and dtype, optionally assigning random gradients, and does the same for
                # direct buffers, without using the module to method under test.
                for c in m.children():
                    _to(c, set_grad=set_grad)
                for n, p in m.named_parameters(recurse=False):
                    new_p = torch.nn.Parameter(p.detach().clone().to(device, dtype))
                    setattr(m, n, new_p)
                    if set_grad:
                        new_p.grad = torch.randn_like(new_p)
                for n, b in m.named_buffers(recurse=False):
                    new_b = b.detach().clone().to(device, dtype)
                    setattr(m, n, new_b)
            _to(m, set_grad=set_grad)

            # Check .to() can be run after forward and backward with swap
            has_params = len(list(m.parameters())) > 0
            if swap and not set_grad and has_params:
                out = m(*args, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]
                out.sum().backward()
                m.to(dtype=torch.half)
                # reset
                m.to(dtype=torch.float32)

            prev_device, prev_dtype = device, dtype
            for device_, dtype_ in product(devices, dtypes):
                # if device/dtype do not change, grad.to(device, dtype) is a no-op so
                # swapping will not change ._cdata
                # parameters will be wrapped in an nn.Parameter before swapping
                # which will cause the ._cdata to change
                g_no_swap = device_ == prev_device and dtype_ == prev_dtype
                prev_device, prev_dtype = device_, dtype_

                p_ids_before = [id(p) for p in m.parameters()]
                p_cdatas_before = [p._cdata for p in m.parameters()]
                if set_grad:
                    g_ids_before = [id(p.grad) for p in m.parameters()]
                    g_cdatas_before = [p.grad._cdata for p in m.parameters()]

                m.to(device=device_, dtype=dtype_)

                self.assertTrue(all(isinstance(p, torch.nn.Parameter) for p in m.parameters()))
                self.assertTrue(all(p.device.type == device_ for p in m.parameters()))
                self.assertTrue(all(p.dtype == dtype_ for p in m.parameters()))
                p_ids_after = [id(p) for p in m.parameters()]
                p_cdatas_after = [p._cdata for p in m.parameters()]

                if set_grad:
                    self.assertTrue(all(p.grad.device.type == device_ for p in m.parameters()))
                    self.assertTrue(all(p.grad.dtype == dtype_ for p in m.parameters()))
                    g_ids_after = [id(p.grad) for p in m.parameters()]
                    g_cdatas_after = [p.grad._cdata for p in m.parameters()]

                if swap:
                    # id same, ._cdata differs --> swapped cdata of THPVariable
                    self.assertTrue(all(a == b for a, b in zip(p_ids_before, p_ids_after)))
                    self.assertTrue(all(a != b for a, b in zip(p_cdatas_before, p_cdatas_after)))
                    if set_grad:
                        self.assertTrue(
                            all(a == b if g_no_swap else a != b for a, b in zip(g_cdatas_before, g_cdatas_after)))
                else:
                    # id and _cdata remain the same --> .data setting
                    self.assertTrue(all(a == b for a, b in zip(p_cdatas_before, p_cdatas_after)))
                    self.assertTrue(all(a == b for a, b in zip(p_ids_before, p_ids_after)))
                    if set_grad:
                        self.assertTrue(all(a == b for a, b in zip(g_cdatas_before, g_cdatas_after)))
                        self.assertTrue(all(a == b for a, b in zip(g_ids_before, g_ids_after)))

    @modules([module for module in module_db if not module.is_lazy], allowed_dtypes=[torch.float32])
    @parametrize('swap', [True, False])
    @wrapSwapTensorsTest()
    def test_to_empty(self, device, dtype, module_info, swap, training):
        # [FUNCTION DOC] Float32 only, not MPS, run for swap on or off. Builds each module inside
        # a meta device context, records parameter ids and underlying handles, calls to_empty
        # with the target device, and asserts all parameters are Parameters on that device with
        # the test dtype. With swap the ids stay the same and the handles change; without swap
        # both change because new Parameter objects are created.
        module_cls = module_info.module_cls

        with torch.device("meta"):
            module_inputs = module_info.module_inputs_func(module_info, device=None, dtype=dtype,
                                                           requires_grad=False, training=training)

        torch.__future__.set_swap_module_params_on_conversion(swap)
        device_ = torch.device(device)

        for module_input in module_inputs:
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            with torch.device("meta"):
                m = module_cls(*c_args, **c_kwargs)

            p_ids_before = [id(p) for p in m.parameters()]
            p_cdatas_before = [p._cdata for p in m.parameters()]
            m.to_empty(device=device_)

            self.assertTrue(all(isinstance(p, torch.nn.Parameter) for p in m.parameters()))
            self.assertTrue(all(p.device == device_ for p in m.parameters()))
            self.assertTrue(all(p.dtype == dtype for p in m.parameters()))
            p_ids_after = [id(p) for p in m.parameters()]
            p_cdatas_after = [p._cdata for p in m.parameters()]

            if swap:
                # id same, ._cdata differs --> swapped cdata of THPVariable
                self.assertTrue(all(a == b for a, b in zip(p_ids_before, p_ids_after)))
                self.assertTrue(all(a != b for a, b in zip(p_cdatas_before, p_cdatas_after)))
            else:
                # id and ._cdata differ
                # meta and device have different shallow copy types, so this will create a new
                # parameter and assign it to the module
                self.assertTrue(all(a != b for a, b in zip(p_ids_before, p_ids_after)))
                self.assertTrue(all(a != b for a, b in zip(p_cdatas_before, p_cdatas_after)))


class TestJitReplaceSubmodule(TestCase):
    # [CLASS DOC] Tests for the C++ binding that replaces a named submodule of a scripted module.
    # Each test scripts small module trees, calls the binding with the underlying C++ module, a
    # dotted path and the replacement C++ module, and checks results or error messages.
    def test_jit_replace_submodule(self):
        # [FUNCTION DOC] Scripts a parent with a child and a replacement child that doubles its
        # output. Replaces the child by name, wraps the returned C++ module, and asserts the
        # output changed and the child C++ type name equals the replacement's.
        from torch.jit._recursive import wrap_cpp_module

        class SubA(torch.nn.Module):
            # [CLASS DOC] HELPER: Child module with a linear layer whose forward returns
            # its output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a linear layer with 4 inputs and 4 outputs as
                # an attribute.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the input.
                return self.linear(x)

        class SubB(torch.nn.Module):
            # [CLASS DOC] HELPER: Replacement child; same layout as SubA but forward doubles
            # the output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a linear layer with 4 inputs and 4 outputs as
                # an attribute.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and multiplies the result
                # by two.
                return self.linear(x) * 2

        class Parent(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module holding a SubA instance as its child attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child attribute holding a SubA instance.
                super().__init__()
                self.child = SubA()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the child.
                return self.child(x)

        root = torch.jit.script(Parent())
        new_child = torch.jit.script(SubB())

        inp = torch.randn(2, 4)
        out_before = root(inp)

        root = wrap_cpp_module(
            torch._C._jit_replace_submodule(root._c, "child", new_child._c)
        )

        out_after = root(inp)
        self.assertFalse(torch.equal(out_before, out_after))
        self.assertEqual(
            root.child._c._type().name(),
            new_child._c._type().name(),
        )

    def test_jit_replace_submodule_mutates_in_place(self):
        # [FUNCTION DOC] Checks the replacement binding mutates the input C++ module in place.
        # The return value is discarded; asserts the original C++ module object is the same
        # instance and that re-wrapping it (to bypass the cached Python children) shows the
        # replacement type. Also asserts beforehand that the two child type names differ.
        # _jit_replace_submodule must mutate the input ``root._c`` in
        # place rather than swap on a clone.  Old behaviour cloned root
        # first, returned the clone, and left ``root._c`` untouched --
        # which silently broke callers that hold onto the original root
        # and discard the return value (e.g. regional-AOTI's
        # create_lowered_from_scripted_merge before the in-place fix).
        #
        # We must NOT inspect via ``root.child`` -- ``RecursiveScriptModule``
        # caches Python child wrappers at construction and would return the
        # stale SubA wrapper regardless of any underlying C++ swap.  Re-wrap
        # ``root._c`` after the call to get a fresh Python view that reflects
        # the current C++ attribute slots; under the in-place contract the
        # re-wrap sees SubB, under the cloned contract it still sees SubA.
        from torch.jit._recursive import wrap_cpp_module

        class SubA(torch.nn.Module):
            # [CLASS DOC] HELPER: Original child module with a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a linear layer with 4 inputs and 4 outputs as
                # an attribute.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the input.
                return self.linear(x)

        class SubB(torch.nn.Module):
            # [CLASS DOC] HELPER: Replacement child; forward doubles the linear output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a linear layer with 4 inputs and 4 outputs as
                # an attribute.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and multiplies the result
                # by two.
                return self.linear(x) * 2

        class Parent(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module holding a SubA instance as its child attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child attribute holding a SubA instance.
                super().__init__()
                self.child = SubA()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the child.
                return self.child(x)

        root = torch.jit.script(Parent())
        new_child = torch.jit.script(SubB())
        root_c_before = root._c
        original_child_type_name = root.child._c._type().name()
        new_child_type_name = new_child._c._type().name()
        self.assertNotEqual(original_child_type_name, new_child_type_name)

        # Intentionally discard the return value so we are checking the
        # mutation on the input ``root._c``.
        torch._C._jit_replace_submodule(root._c, "child", new_child._c)

        # Underlying C++ ScriptModule object must be the same instance --
        # the in-place contract returns the input, not a clone.
        self.assertIs(root._c, root_c_before)

        # Re-wrap to bypass RecursiveScriptModule's cached children dict
        # and read the current state of ``root._c``'s attribute slots.
        re_wrapped = wrap_cpp_module(root._c)
        self.assertEqual(
            re_wrapped.child._c._type().name(),
            new_child_type_name,
        )

    def test_jit_replace_submodule_nested(self):
        # [FUNCTION DOC] Replaces a leaf two levels down using a dotted path to the leaf, with
        # a scripted leaf of scale 3. Asserts the whole tree then outputs the input multiplied
        # by 3.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Leaf module storing a float scale and multiplying its input
            # by it.
            def __init__(self, scale: float) -> None:
                # [FUNCTION DOC] HELPER: Stores the scale as an attribute.
                super().__init__()
                self.scale = scale

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by the scale.
                return x * self.scale

        class Mid(torch.nn.Module):
            # [CLASS DOC] HELPER: Middle module holding a Leaf with scale 1.0.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the leaf attribute with scale 1.0.
                super().__init__()
                self.leaf = Leaf(1.0)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the leaf.
                return self.leaf(x)

        class Root(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module holding a Mid instance.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the mid attribute.
                super().__init__()
                self.mid = Mid()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the mid module.
                return self.mid(x)

        root = torch.jit.script(Root())
        new_leaf = torch.jit.script(Leaf(3.0))

        inp = torch.randn(2, 4)
        root = torch.jit._recursive.wrap_cpp_module(
            torch._C._jit_replace_submodule(root._c, "mid.leaf", new_leaf._c)
        )

        out_after = root(inp)
        self.assertEqual(out_after, inp * 3.0)

    def test_jit_replace_submodule_empty_qualified_name(self):
        # [FUNCTION DOC] Asserts an empty qualified name makes the binding raise a RuntimeError
        # whose message mentions needing a non-empty qualified name.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity leaf module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return x

        class Root(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module holding an identity Leaf as its child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child attribute.
                super().__init__()
                self.child = Leaf()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the child.
                return self.child(x)

        root = torch.jit.script(Root())
        new_child = torch.jit.script(Leaf())

        with self.assertRaisesRegex(RuntimeError, "non-empty qualified_name"):
            torch._C._jit_replace_submodule(root._c, "", new_child._c)

    def test_jit_replace_submodule_missing_segment(self):
        # [FUNCTION DOC] Asserts that a path whose first segment names a missing submodule raises
        # a RuntimeError saying the path segment could not be found, both for a two-segment path
        # and for a single missing name.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity leaf module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return x

        class Root(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module holding an identity Leaf as its child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child attribute.
                super().__init__()
                self.child = Leaf()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the child.
                return self.child(x)

        root = torch.jit.script(Root())
        new_child = torch.jit.script(Leaf())

        with self.assertRaisesRegex(RuntimeError, "could not find submodule path segment 'missing'"):
            torch._C._jit_replace_submodule(root._c, "missing.leaf", new_child._c)

        with self.assertRaisesRegex(RuntimeError, "could not find submodule path segment 'missing'"):
            torch._C._jit_replace_submodule(root._c, "missing", new_child._c)

    def test_jit_replace_submodule_non_module_attribute(self):
        # [FUNCTION DOC] Asserts that naming an integer attribute, alone or as the start of a
        # longer path, raises a RuntimeError saying that attribute is not a submodule.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity leaf module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return x

        class Root(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module with a child Leaf and an integer attribute set to 7
            # (class level annotation so scripting treats it as an int).
            scalar: int

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child and sets the integer attribute to 7.
                super().__init__()
                self.child = Leaf()
                self.scalar = 7

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the child output plus the integer attribute.
                return self.child(x) + self.scalar

        root = torch.jit.script(Root())
        new_child = torch.jit.script(Leaf())

        with self.assertRaisesRegex(RuntimeError, "'scalar'.*is not a submodule"):
            torch._C._jit_replace_submodule(root._c, "scalar", new_child._c)

        with self.assertRaisesRegex(RuntimeError, "'scalar'.*is not a submodule"):
            torch._C._jit_replace_submodule(root._c, "scalar.nested", new_child._c)

    def test_jit_replace_submodule_shared_parent_type(self):
        # [FUNCTION DOC] Builds a root with two Mid instances of the same class and tries to
        # replace the leaf under one of them. Asserts an exception mentioning unique parent
        # types, because the two parents share a scripted type.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Leaf module storing a float scale and multiplying its input
            # by it.
            def __init__(self, scale: float) -> None:
                # [FUNCTION DOC] HELPER: Stores the scale as an attribute.
                super().__init__()
                self.scale = scale

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by the scale.
                return x * self.scale

        class Mid(torch.nn.Module):
            # [CLASS DOC] HELPER: Middle module holding a Leaf with scale 1.0.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the leaf attribute with scale 1.0.
                super().__init__()
                self.leaf = Leaf(1.0)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the leaf.
                return self.leaf(x)

        class Root(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module holding two separate Mid instances of the
            # same class.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two Mid attributes.
                super().__init__()
                self.mid_a = Mid()
                self.mid_b = Mid()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of both Mid outputs.
                return self.mid_a(x) + self.mid_b(x)

        root = torch.jit.script(Root())
        new_leaf = torch.jit.script(Leaf(2.0))

        with self.assertRaisesRegex(Exception, "unique parent types"):
            torch._C._jit_replace_submodule(root._c, "mid_a.leaf", new_leaf._c)

    def test_jit_replace_submodule_empty_path_segment(self):
        # [FUNCTION DOC] Asserts a path containing an empty segment between two separators raises
        # a RuntimeError mentioning empty path segments.
        class Leaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity leaf module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return x

        class Mid(torch.nn.Module):
            # [CLASS DOC] HELPER: Middle module holding an identity Leaf.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the leaf attribute.
                super().__init__()
                self.leaf = Leaf()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the leaf.
                return self.leaf(x)

        class Root(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module holding a Mid instance.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the mid attribute.
                super().__init__()
                self.mid = Mid()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the mid module.
                return self.mid(x)

        root = torch.jit.script(Root())
        new_leaf = torch.jit.script(Leaf())

        with self.assertRaisesRegex(RuntimeError, "empty path segments"):
            torch._C._jit_replace_submodule(root._c, "mid..leaf", new_leaf._c)


instantiate_device_type_tests(TestModule, globals(), allow_mps=True, allow_xpu=True)

if __name__ == '__main__':
    run_tests()
