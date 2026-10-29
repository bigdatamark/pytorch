# mypy: ignore-errors

# [MODULE DOC] ModuleInfo database for the generic nn.Module tests (the TestModule suites in
# the test files).
# Defines the data classes used to describe test samples (FunctionInput, ModuleInput,
# ErrorModuleInput, ModuleErrorEnum), the ModuleInfo metadata record, and the modules decorator
# that parametrizes a device-generic test over ModuleInfo entries, train/eval mode and dtype.
# Contains one sample-generator function per tested module (module_inputs_torch_nn_*),
# error-sample generators (module_error_inputs_torch_nn_*), no-batch-dim reference helpers, shared
# decorator lists for RNN-family modules, and the module_db list that ties them together with
# per-module skips and tolerances.
# Import-time side effects: builds MODULE_CLASSES from the public names of the listed nn
# namespaces and fills MODULE_CLASS_NAMES (first namespace wins for aliases); queries the MPS
# backend for the macOS version; constructs every ModuleInfo in module_db. No tensors are
# created at import.
# KNOWN ISSUE: MODULE_NAMESPACES lists the quantized modules namespace twice and MODULES_TO_SKIP
# lists the quantized MaxPool2d twice; the duplicates are harmless.

import torch
import unittest
from copy import deepcopy
from enum import Enum
from functools import wraps, partial
from itertools import chain, product
import itertools
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_dtype import (
    floating_types, floating_and_complex_types_and, get_all_fp_dtypes)
from torch.testing._internal.common_device_type import (
    _TestParametrizer, _update_param_kwargs, expectedFailureMPS, toleranceOverride, tol,
    precisionOverride, skipMeta, skipMPS)
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_nn import (
    cosineembeddingloss_reference, cross_entropy_loss_reference, ctcloss_reference,
    hingeembeddingloss_reference, huberloss_reference, kldivloss_reference, linear_cross_entropy_loss_reference,
    marginrankingloss_reference, multimarginloss_reference, multilabelmarginloss_reference,
    nllloss_reference, nlllossNd_reference, smoothl1loss_reference, softmarginloss_reference, get_reduction)
from torch.testing._internal.common_utils import (
    freeze_rng_state, skipIfMPS, GRADCHECK_NONDET_TOL, TEST_WITH_ROCM, IS_WINDOWS,
    skipIfTorchDynamo, skipIfXpu)
from types import ModuleType
import operator

# List of all namespaces containing modules to test.
MODULE_NAMESPACES: list[ModuleType] = [
    torch.nn.modules,
    torch.ao.nn.qat.modules,
    torch.ao.nn.quantizable.modules,
    torch.ao.nn.quantized.modules,
    torch.ao.nn.quantized.modules,
]

# Modules that shouldn't be tested for one reason or another.
MODULES_TO_SKIP: set[type] = {
    torch.nn.Module,  # abstract base class
    torch.nn.Container,  # deprecated
    torch.nn.NLLLoss2d,  # deprecated
    torch.ao.nn.quantized.MaxPool2d,  # aliases to nn.MaxPool2d
    torch.ao.nn.quantized.MaxPool2d,  # aliases to nn.MaxPool2d
}

# List of all module classes to test.
MODULE_CLASSES: list[type] = [*chain.from_iterable([
    [getattr(namespace, module_name) for module_name in namespace.__all__]  # type: ignore[attr-defined]
    for namespace in MODULE_NAMESPACES])]
MODULE_CLASSES = [cls for cls in MODULE_CLASSES if cls not in MODULES_TO_SKIP]

# Dict of module class -> common name. Useful for making test names more intuitive.
# Example: torch.nn.modules.linear.Linear -> "nn.Linear"
MODULE_CLASS_NAMES: dict[type, str] = {}
for namespace in MODULE_NAMESPACES:
    for module_name in namespace.__all__:  # type: ignore[attr-defined]
        module_cls = getattr(namespace, module_name)
        namespace_name = namespace.__name__.replace('torch.', '').replace('.modules', '')

        # Deal with any aliases by preferring earlier names.
        if module_cls not in MODULE_CLASS_NAMES:
            MODULE_CLASS_NAMES[module_cls] = f'{namespace_name}.{module_name}'


# Specifies the modes (i.e. train, eval) to test over.
TrainEvalMode = Enum('TrainEvalMode', ('train_only', 'eval_only', 'train_and_eval'))


class modules(_TestParametrizer):
    # [CLASS DOC] Decorator class (a test parametrizer) that expands a device-generic test into
    # one test per ModuleInfo, training/eval mode and dtype.
    # State: module_info_list (list of ModuleInfo), allowed_dtypes (set or None meaning all
    # supported), train_eval_mode (a TrainEvalMode value) and skip_if_dynamo (bool).
    # Only valid inside device-specific test classes; the generated test receives module_info,
    # dtype and training keyword arguments.
    """ PROTOTYPE: Decorator for specifying a list of modules over which to run a test. """

    def __init__(self, module_info_iterable, allowed_dtypes=None,
                 train_eval_mode=TrainEvalMode.train_and_eval, skip_if_dynamo=True):
        # [FUNCTION DOC] Stores the module infos as a list, the allowed dtypes as a set (or
        # None), the train/eval mode selector and the skip-if-dynamo flag. No validation and
        # no side effects.
        self.module_info_list = list(module_info_iterable)
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None
        self.train_eval_mode = train_eval_mode
        self.skip_if_dynamo = skip_if_dynamo

    def _get_training_flags(self, module_info):
        # [FUNCTION DOC] HELPER: Returns the list of training flags to test for a ModuleInfo: true
        # for train-only or both modes, false for eval-only or both modes. If the module's train
        # and eval behavior do not differ, only the first flag is kept. Pure function.
        training_flags = []
        if (self.train_eval_mode == TrainEvalMode.train_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(True)

        if (self.train_eval_mode == TrainEvalMode.eval_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(False)

        # If train and eval modes don't differ for the module, don't bother using more than one.
        if not module_info.train_and_eval_differ:
            training_flags = training_flags[:1]

        return training_flags

    def _parametrize_test(self, test, generic_cls, device_cls):
        # [FUNCTION DOC] Generator used by the parametrization machinery: for every ModuleInfo,
        # training flag and dtype (supported dtypes for the device type, intersected with
        # allowed_dtypes) it yields a tuple of wrapped test, test name, parameter kwargs
        # (module_info, dtype, training) and a decorator factory bound to the ModuleInfo.
        # The name is the formatted module name, with a train_mode or eval_mode suffix when both
        # modes are generated. Unless skip_if_dynamo is false or the inductor test flag is on, the
        # test is wrapped to skip under Dynamo.
        # Raises RuntimeError if device_cls is None (decorator used outside device-type
        # instantiation). On any error while building a test it prints a message naming the test
        # and module, then re-raises.
        # KNOWN ISSUE: dtypes are held in a set, so the generation order of dtype variants
        # is not deterministic.
        if device_cls is None:
            raise RuntimeError('The @modules decorator is only intended to be used in a device-specific '
                               'context; use it with instantiate_device_type_tests() instead of '
                               'instantiate_parametrized_tests()')

        for module_info in self.module_info_list:
            dtypes = set(module_info.supported_dtypes(device_cls.device_type))
            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            training_flags = self._get_training_flags(module_info)
            for (training, dtype) in product(training_flags, dtypes):
                # Construct the test name; device / dtype parts are handled outside.
                # See [Note: device and dtype suffix placement]
                test_name = module_info.formatted_name
                if len(training_flags) > 1:
                    test_name += f"_{'train_mode' if training else 'eval_mode'}"

                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'module_info': module_info}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)
                _update_param_kwargs(param_kwargs, 'training', training)

                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        # [FUNCTION DOC] HELPER: Thin forwarding wrapper that calls the original
                        # test with the same arguments; it exists so the functools wrapper can be
                        # given its own decorators without mutating the original test.
                        return test(*args, **kwargs)

                    if self.skip_if_dynamo and not torch.testing._internal.common_utils.TEST_WITH_TORCHINDUCTOR:
                        test_wrapper = skipIfTorchDynamo("Policy: we don't run ModuleInfo tests w/ Dynamo")(test_wrapper)

                    decorator_fn = partial(module_info.get_decorators, generic_cls.__name__,
                                           test.__name__, device_cls.device_type, dtype)

                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print(f"Failed to instantiate {test_name} for module {module_info.name}!")
                    raise ex


def get_module_common_name(module_cls):
    # [FUNCTION DOC] Returns the display name for a module class: the precomputed short dotted
    # name such as nn.Linear from MODULE_CLASS_NAMES if present, otherwise the class's own
    # name. Pure lookup.
    if module_cls in MODULE_CLASS_NAMES:
        # Example: "nn.Linear"
        return MODULE_CLASS_NAMES[module_cls]
    else:
        return module_cls.__name__


class FunctionInput:
    # [CLASS DOC] Plain container (slots only) for the positional and keyword arguments of
    # one function call.
    # Attributes: args (tuple) and kwargs (dict). Used for both module constructor input
    # and forward input.
    """ Contains args and kwargs to pass as input to a function. """
    __slots__ = ['args', 'kwargs']

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Captures any positional arguments as the args tuple and any keyword
        # arguments as the kwargs dict, without copying.
        self.args = args
        self.kwargs = kwargs


class ModuleInput:
    # [CLASS DOC] One test sample for a module: constructor arguments, forward arguments, a
    # description suffix and an optional reference function.
    # Attributes: constructor_input and forward_input (FunctionInput; forward may be None), desc
    # (string appended to subtest names), reference_fn (None or a wrapper taking the module plus
    # forward arguments and returning the expected output).
    # Invariant: when a reference is given it is replaced by a wrapper that deep-copies the
    # arguments and passes the module's parameter list as an extra argument.
    """ Contains args / kwargs for module instantiation + forward pass. """
    __slots__ = ['constructor_input', 'forward_input', 'desc', 'reference_fn']

    def __init__(self, constructor_input, forward_input=None, desc='', reference_fn=None):
        # [FUNCTION DOC] Stores the inputs, description and reference. If a reference function is
        # supplied, replaces it with a wrapper that deep-copies the positional and keyword
        # arguments (to avoid side effects) and calls the reference with the module, the list of
        # module parameters and the copied arguments. No other side effects.
        self.constructor_input = constructor_input  # Inputs to pass during construction
        self.forward_input = forward_input  # Inputs to pass to forward()
        self.desc = desc  # Description for this set of inputs
        self.reference_fn = reference_fn  # Reference with signature: reference_fn(module, parameters, *args, **kwargs)

        if reference_fn is not None:

            @wraps(reference_fn)
            def copy_reference_fn(m, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper around the user reference: deep-copies the
                # arguments, then returns the reference's result called with the module, a list of
                # its parameters and the copied arguments.
                # Copy inputs to avoid undesired side effects from calling the reference.
                args, kwargs = deepcopy(args), deepcopy(kwargs)

                # Note that module parameters are passed in for convenience.
                return reference_fn(m, list(m.parameters()), *args, **kwargs)

            self.reference_fn = copy_reference_fn

class ModuleErrorEnum(Enum):
    # [CLASS DOC] Enumeration of the stage at which a module sample is expected to raise:
    # CONSTRUCTION_ERROR (building the module) or FORWARD_ERROR (calling it).
    """ Enumerates when error is raised when testing modules. """
    CONSTRUCTION_ERROR = 0
    FORWARD_ERROR = 1

class ErrorModuleInput:
    # [CLASS DOC] A ModuleInput that is expected to raise, together with the
    # expected failure details.
    # Attributes (slots): module_error_input (the ModuleInput), error_on (a ModuleErrorEnum
    # value), error_type (exception class) and error_regex (pattern the message must match).
    """
    A ModuleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """

    __slots__ = ["module_error_input", "error_on", "error_type", "error_regex"]

    def __init__(self,
                 module_error_input,
                 *,
                 error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
                 error_type=RuntimeError,
                 error_regex):
        # [FUNCTION DOC] Stores the sample and the expectation. error_on defaults to
        # construction-time failure and error_type to RuntimeError; error_regex is
        # keyword-only and required.
        self.module_error_input = module_error_input
        self.error_on = error_on
        self.error_type = error_type
        self.error_regex = error_regex


class ModuleInfo:
    # [CLASS DOC] Metadata record describing how a torch.nn module class is tested: its sample
    # generators, supported dtypes per device, decorators and tolerances.
    # Attributes mirror the constructor arguments (module_cls, module_inputs_func,
    # module_error_inputs_func, decorators combined with skips, dtypes, dtypesIfMPS, dtypesIfHpu,
    # supports_gradgrad, gradcheck_nondet_tol, module_memformat_affects_out,
    # train_and_eval_differ, gradcheck_fast_mode) plus is_lazy, true when the class derives from
    # the lazy-module mixin.
    # Key methods: get_decorators, supported_dtypes and the name properties. Instances are
    # collected in module_db.
    """ Module information to be used in testing. """

    def __init__(self,
                 module_cls,  # Class object for the module under test
                 *,
                 module_inputs_func,  # Function to generate module inputs
                 skips=(),  # Indicates which tests to skip
                 decorators=None,  # Additional decorators to apply to generated tests
                 dtypes=floating_types(),  # dtypes this function is expected to work with
                 dtypesIfMPS=(torch.float16, torch.float32,),  # dtypes this function is expected to work with on MPS
                 dtypesIfHpu=(torch.bfloat16, torch.float32,),
                 supports_gradgrad=True,  # whether the op supports second order gradients
                 gradcheck_nondet_tol=0.0,  # tolerance for nondeterminism while performing gradcheck
                 module_memformat_affects_out=False,  # whether converting module to channels last will generate
                                                      # channels last output
                 train_and_eval_differ=False,  # whether the module has differing behavior between train and eval
                 module_error_inputs_func=None,  # Function to generate module inputs that error
                 gradcheck_fast_mode=None,  # Whether to use the fast implementation for gradcheck/gradgradcheck.
                                            # When set to None, defers to the default value provided by the wrapper
                                            # function around gradcheck (testing._internal.common_utils.gradcheck)
                 ):
        # [FUNCTION DOC] Records the module class and testing options. The decorators and skips
        # are concatenated into one tuple (decorators first). Keyword-only options default to
        # floating-point dtypes, a fixed reduced dtype set for MPS and a bfloat16/float32 set for
        # HPU, gradgrad support on, zero nondeterminism tolerance and no error generator. Computes
        # is_lazy from the class hierarchy.
        self.module_cls = module_cls
        self.module_inputs_func = module_inputs_func
        self.decorators = (*(decorators if decorators else []), *(skips if skips else []))
        self.dtypes = dtypes
        self.dtypesIfMPS = dtypesIfMPS
        self.dtypesIfHpu = dtypesIfHpu
        self.supports_gradgrad = supports_gradgrad
        self.gradcheck_nondet_tol = gradcheck_nondet_tol
        self.module_memformat_affects_out = module_memformat_affects_out
        self.train_and_eval_differ = train_and_eval_differ
        self.module_error_inputs_func = module_error_inputs_func
        self.gradcheck_fast_mode = gradcheck_fast_mode
        self.is_lazy = issubclass(module_cls, torch.nn.modules.lazy.LazyModuleMixin)

    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        # [FUNCTION DOC] Returns the flat list of decorators that apply to a test: DecorateInfo
        # entries contribute their decorators only when active for the given test class, test
        # name, device, dtype and parameter kwargs; any other entry is included unconditionally.
        result = []
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(test_class, test_name, device, dtype, param_kwargs):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result

    def supported_dtypes(self, device_type):
        # [FUNCTION DOC] Returns the dtype collection to test for a device type: the MPS list for
        # mps, the HPU list for hpu, otherwise the default dtypes.
        if device_type == 'mps':
            return self.dtypesIfMPS
        elif device_type == 'hpu':
            return self.dtypesIfHpu
        else:
            return self.dtypes

    @property
    def name(self):
        # [FUNCTION DOC] Property returning the common display name of the module class,
        # such as nn.Linear.
        return get_module_common_name(self.module_cls)

    @property
    def formatted_name(self):
        # [FUNCTION DOC] Property returning the name with dots replaced by underscores, suitable
        # for building test names.
        return self.name.replace('.', '_')

# Start of module inputs functions.

def module_inputs_torch_nn_Linear(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Linear: batched input with bias, batched
    # without bias, and an unbatched 1D input; each has an explicit reference computed from
    # the module parameters.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(10, 8),
                    forward_input=FunctionInput(input=make_input((4, 10))),
                    reference_fn=lambda m, p, input: torch.mm(input, p[0].t()) + p[1].view(1, -1).expand(4, 8)),
        ModuleInput(constructor_input=FunctionInput(10, 8, bias=False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='no_bias',
                    reference_fn=lambda m, p, i: torch.mm(i, p[0].t())),
        ModuleInput(constructor_input=FunctionInput(3, 5),
                    forward_input=FunctionInput(make_input(3)),
                    desc='no_batch_dim',
                    reference_fn=lambda m, p, i: torch.mm(i.view(1, -1), p[0].t()).view(-1) + p[1])
    ]

    return module_inputs


def module_inputs_torch_nn_Bilinear(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Bilinear: batched with bias, without bias, and
    # unbatched inputs; the reference is computed with an einsum over the two inputs and
    # the module parameters.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def bilinear_reference_fn(m, p, x1, x2, bias=True):
        # [FUNCTION DOC] HELPER: Computes the expected bilinear output for batched inputs with an
        # einsum over the weight, then adds the bias when enabled (viewed as a vector for batch
        # size one, otherwise broadcast across the batch).
        result = torch.einsum('bn,anm,bm->ba', x1, p[0], x2)
        if bias:
            if x1.shape[0] == 1:
                result = result.view(-1) + p[1]
            else:
                result = result + p[1].view(1, -1).expand(x1.shape[0], p[0].shape[0])
        return result

    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(2, 3, 4),
                    forward_input=FunctionInput(make_input((8, 2)), make_input((8, 3))),
                    reference_fn=bilinear_reference_fn),
        ModuleInput(constructor_input=FunctionInput(2, 3, 4, bias=False),
                    forward_input=FunctionInput(make_input((8, 2)), make_input((8, 3))),
                    desc='no_bias',
                    reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1, x2, bias=False)),
        ModuleInput(constructor_input=FunctionInput(2, 3, 4),
                    forward_input=FunctionInput(make_input(2), make_input(3)),
                    desc='no_batch_dim',
                    reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1.view(1, -1), x2.view(1, -1))),
    ]

    return module_inputs


def module_inputs_torch_nn_KLDivLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.KLDivLoss: cases for default, sum, batchmean and none
    # reduction plus log_target, each on a 10x10 input and on a scalar input; the reference is
    # kldivloss_reference. Reads an optional log_target keyword to decide whether the target is
    # built in log space.
    # KNOWN ISSUE: scalar_target is computed but never used (a FIXME notes this); the scalar
    # samples pass the scalar input as both input and target. The log_target case sets the
    # constructor option but the target is built from the keyword argument, so it stays log-space
    # unless that keyword is passed.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_batchmean', {'reduction': 'batchmean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('log_target', {'log_target': True})
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of kldivloss_reference for the given input(s) and the bound case
            # options. The case options are bound as defaults so each sample keeps its own.
            return kldivloss_reference(i, t, **constructor_kwargs)

        input = make_input((10, 10)).log()
        target = make_input((10, 10)) if kwargs.get('log_target', False) else make_input((10, 10)).log()
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(input, target),
                        desc=desc,
                        reference_fn=reference_fn)
        )

        scalar_input = make_input(()).log()
        # FIXME(rec): scalar_target is unused, perhaps should be argument to FunctionInput?
        scalar_target = (  # noqa: F841
            make_input(()) if kwargs.get('log_target', False) else make_input(()).log()
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(scalar_input, scalar_input),
                        desc='scalar_' + desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_NLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.NLLLoss: cases for default, sum/none reduction,
    # ignore_index (including negative) and class weights; each case is run on a 2D (15x4)
    # input, a 4D spatial input, a 6D higher-dimensional input and a 3D input. Inputs are
    # log-softmaxed over dim 1 and targets are random class indices below 4; references are
    # nllloss_reference and nlllossNd_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    def make_input(shape, device=device, dtype=dtype, requires_grad=requires_grad):
        # [FUNCTION DOC] HELPER: Creates a random tensor of the given shape, log-softmaxed over
        # dim 1, with requires_grad applied after the softmax so it is a leaf.
        return make_tensor(shape, device=device, dtype=dtype,
                           requires_grad=False).log_softmax(dim=1).requires_grad_(requires_grad)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_none', {'reduction': 'none'}),
        ('ignore_index', {'ignore_index': 2}),
        ('weights', {'weight': make_weight(4).abs()}),
        ('weights_ignore_index', {'weight': make_weight(4).abs(), 'ignore_index': 2}),
        ('weights_ignore_index_neg', {'weight': make_weight(4).abs(), 'ignore_index': -1})
    ]

    # TODO: Uncomment when negative weights is supported.
    # negative_weight = make_weight(10)
    # negative_weight[0] = -1
    # cases.append(('weights_negative', {'weight': negative_weight}))
    module_inputs = []
    for desc, constructor_kwargs in cases:

        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of nllloss_reference for the given input(s) and the bound case
            # options. The case options are bound as defaults so each sample keeps its own.
            return nllloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 4)),
                                                    torch.empty(15, device=device).uniform_().mul(4).floor().long()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

        def nd_reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference for the spatial and
            # higher-dimensional cases: returns nlllossNd_reference for the input and
            # target with the bound case options.
            return nlllossNd_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5, 5)),
                            torch.empty(2, 5, 5, device=device).uniform_().mul(4).floor().long()),
                        desc=f"nd_{desc}",
                        reference_fn=nd_reference_fn)
        )

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5, 5, 2, 2)),
                            torch.empty(2, 5, 5, 2, 2, device=device).uniform_().mul(4).floor().long()),
                        desc=f"higher_dim_{desc}",
                        reference_fn=nd_reference_fn)
        )

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5)),
                            torch.empty(2, 5, device=device).uniform_().mul(4).floor().long()),
                        desc=f"3d_{desc}",
                        reference_fn=nd_reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_GaussianNLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.GaussianNLLLoss: cases for default, sum, mean and
    # none reduction plus a homoscedastic variant (variance of shape 1x3, otherwise 4x1 with
    # absolute values) on 4x3 inputs; the reference is the generic no-batch-dim reference. The
    # homoscedastic flag is removed from the constructor options before use.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('homoscedastic', {'homoscedastic': True}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        homoscedastic = constructor_kwargs.pop('homoscedastic', False)
        var_input = make_input(1, 3).abs() if homoscedastic else make_input(4, 1).abs()
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(4, 3),
                                                    make_target(4, 3),
                                                    var_input),
                        desc=desc,
                        reference_fn=no_batch_dim_reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_PoissonNLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.PoissonNLLLoss: cases for the reductions, full, no
    # log_input and both; input and target are 2x3x4x5 (targets are non-negative whole numbers).
    # For float16 the value range is narrowed to avoid overflow in the reference exponential; a
    # nested reference function computes the expected loss.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    # Narrow the input range for fp16 so `i.exp()` summed over numel doesn't
    # blow past 65504 in the `full` reference path; the default (-9, 9) range
    # gives `exp(8.9) ~= 7300` and over 120 elements overflows fp16 to inf,
    # while the fused kernel accumulates in higher precision and stays finite.
    fp16_kwargs = {'low': -2, 'high': 2} if dtype == torch.float16 else {}
    make_input = partial(make_tensor, device=device, dtype=dtype,
                         requires_grad=requires_grad, **fp16_kwargs)
    make_target = partial(make_tensor, device=device, dtype=dtype,
                          requires_grad=False, **fp16_kwargs)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('full', {'full': True}),
        ('no_log_input', {'log_input': False}),
        ('full_no_log_input', {'full': True, 'log_input': False}),
    ]

    def poissonnllloss_reference_fn(i, t, log_input=True, full=False, reduction='mean', eps=1e-8):
        # [FUNCTION DOC] HELPER: Expected Poisson negative log-likelihood: exponential form when
        # log_input else logarithm with eps; when full, adds the Stirling term masked to zero
        # where the target is at most one; then returns the elementwise result, its mean (sum over
        # element count) or its sum per the reduction.
        if log_input:
            result = i.exp() - t.mul(i)
        else:
            result = i - t.mul((i + eps).log())

        if full:
            result += (t.mul(t.log()) - t + 0.5 * (2. * math.pi * t).log()).masked_fill(t <= 1, 0)

        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()
        else:
            return result.sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters
            # and returns the result of the nested poisson reference for the given input(s) and
            # the bound case options. The case options are bound as defaults so each sample
            # keeps its own.
            return poissonnllloss_reference_fn(i, t, **constructor_kwargs)

        log_input = constructor_kwargs.get('log_input', True)
        input = make_input((2, 3, 4, 5)) if log_input else make_input((2, 3, 4, 5)).abs().add(0.001)
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(input,
                                                    make_target((2, 3, 4, 5)).floor_().abs_()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_MSELoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MSELoss: cases for default, sum, mean and none
    # reduction, each on a 2x3x4x5 input and on a scalar input; a nested reference computes
    # squared error with the matching reduction.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    def mse_loss_reference_fn(m, p, i, t, reduction='mean'):
        # [FUNCTION DOC] HELPER: Expected squared error: elementwise for reduction none, the sum
        # divided by element count for mean, otherwise the sum.
        if reduction == 'none':
            return (i - t).pow(2)
        elif reduction == 'mean':
            return (i - t).pow(2).sum() / i.numel()
        else:
            return (i - t).pow(2).sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 4, 5)),
                                                    make_target((2, 3, 4, 5))),
                        desc=desc,
                        reference_fn=partial(mse_loss_reference_fn, **constructor_kwargs))
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(()),
                                                    make_target(())),
                        desc=f'{desc}_scalar',
                        reference_fn=partial(mse_loss_reference_fn, **constructor_kwargs))
        )

    return module_inputs


def no_batch_dim_reference_fn(m, p, *args, **kwargs):
    # [FUNCTION DOC] Reference implementation used to check that a module accepts unbatched
    # input: each positional tensor gets a batch dimension added (dim 0 when batch_first, which
    # defaults to true, else dim 1), the module is run under frozen RNG state, and the output has
    # that dimension removed.
    # Bound keyword options (removed before calling the module): batch_first, kwargs_to_batchify
    # (mapping from keyword name to the dim at which that tensor argument gets a batch dimension)
    # and is_criterion (when true and the module's reduction is none, one more leading dimension
    # is squeezed from the result). Only modules returning a single tensor are supported.
    # Raises AssertionError if kwargs_to_batchify is given but is not a dict. First two arguments
    # (module, parameters) are accepted and ignored for the parameters.
    """Reference function for modules supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.

    Currently it only supports modules which return a single Tensor as output.
    You can bind the following kwargs.
    Kwargs:
        batch_first[bool] : If True, all the Tensors in `args` while be unsqueezed at dim `0` .
                        and output will be squeezed at dim `0` else dim `1` for both.
        kwargs_to_batchify[dict] : Dictionary specifying the name of the argument and dimension to unsqueeze.
                               Useful if there are few arguments whose batch dimension are different
                               from the ones selected by `batch_first`.
        is_criterion[bool] : Specify if the module is a criterion and handle the reduction for output accordingly.
    """
    def get_and_pop(key, default):
        # [FUNCTION DOC] HELPER: Returns the value of a keyword argument (or the default
        # when absent) and removes it from the enclosing keyword dictionary so it is not
        # passed to the module.
        v = kwargs.get(key, default)
        if key in kwargs:
            kwargs.pop(key)
        return v

    batch_dim = 0 if get_and_pop('batch_first', True) else 1
    kwargs_to_batchify = get_and_pop('kwargs_to_batchify', None)
    is_criterion = get_and_pop('is_criterion', False)

    if kwargs_to_batchify is not None:
        if not isinstance(kwargs_to_batchify, dict):
            raise AssertionError(f"Expected kwargs_to_batchify to be a dict, got {type(kwargs_to_batchify)}")
        for k, v in kwargs.items():
            if k in kwargs_to_batchify and v is not None:
                bdim = kwargs_to_batchify[k]
                kwargs[k] = v.unsqueeze(bdim)

    single_batch_input_args = [input.unsqueeze(batch_dim) for input in args]
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs).squeeze(batch_dim)

    if is_criterion:
        reduction = get_reduction(m)
        if reduction == 'none':
            return output.squeeze(0)
    return output


def no_batch_dim_reference_mha(m, p, *args, **kwargs):
    # [FUNCTION DOC] Reference for MultiheadAttention with unbatched input: adds a batch
    # dimension to each positional tensor (dim 0 unless batch_first is false, which is removed
    # from keywords) and to a non-None key_padding_mask, runs the module under frozen RNG state,
    # and returns the output squeezed at the batch dim together with the attention weights
    # squeezed at dim 0.
    """Reference function for MultiheadAttention supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    batch_dim = 0 if kwargs.get('batch_first', True) else 1
    if 'batch_first' in kwargs:
        kwargs.pop('batch_first')
    if 'key_padding_mask' in kwargs and kwargs['key_padding_mask'] is not None:
        kwargs['key_padding_mask'] = kwargs['key_padding_mask'].unsqueeze(0)
    single_batch_input_args = [input.unsqueeze(batch_dim) for input in args]
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), output[1].squeeze(0))


def no_batch_dim_reference_rnn_gru(m, p, *args, **kwargs):
    # [FUNCTION DOC] Reference for RNN and GRU with unbatched input. Accepts the input alone or
    # the input plus an initial hidden state (given a batch dim at dim 1), adds a batch dim to the
    # input according to the required batch_first keyword (removed before the call), runs the
    # module under frozen RNG state and returns the output squeezed at the batch dim and the final
    # hidden state squeezed at dim 1.
    # Raises KeyError if batch_first is missing.
    # KNOWN ISSUE: with any other positional argument count the input variable is never bound
    # (UnboundLocalError); also packed-sequence samples use this reference although it calls
    # tensor-only methods on the input.
    """Reference function for RNN and GRU supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    if len(args) == 1:
        inp, = args
        h = None
    elif len(args) == 2:
        inp, h = args
        h = h.unsqueeze(1)

    batch_dim = 0 if kwargs['batch_first'] else 1
    kwargs.pop('batch_first')
    inp = inp.unsqueeze(batch_dim)
    single_batch_input_args = (inp, h)
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), output[1].squeeze(1))


def no_batch_dim_reference_lstm(m, p, *args, **kwargs):
    # [FUNCTION DOC] Reference for LSTM with unbatched input. Accepts the input alone or with an
    # initial (hidden, cell) pair (each given a batch dim at dim 1), adds a batch dim to the input
    # per the required batch_first keyword (removed before the call), runs the module under frozen
    # RNG state and returns the output squeezed at the batch dim plus the hidden and cell states
    # squeezed at dim 1.
    # Raises KeyError if batch_first is missing; other positional argument counts leave the
    # input unbound (UnboundLocalError).
    """Reference function for LSTM supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    if len(args) == 1:
        inp, = args
        h = None
    elif len(args) == 2:
        inp, h = args
        h = (h[0].unsqueeze(1), h[1].unsqueeze(1))

    batch_dim = 0 if kwargs['batch_first'] else 1
    kwargs.pop('batch_first')
    inp = inp.unsqueeze(batch_dim)
    single_batch_input_args = (inp, h)
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), (output[1][0].squeeze(1), output[1][1].squeeze(1)))


def no_batch_dim_reference_lstmcell(m, p, *args, **kwargs):
    # [FUNCTION DOC] Reference for LSTMCell with unbatched input: takes the input and a (hidden,
    # cell) pair, adds a leading batch dim to all three, runs the module under frozen RNG state
    # and returns the two outputs with the batch dim removed.
    """Reference function for LSTMCell supporting no batch dimensions.

    The module is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    inp, (h, c) = args
    single_batch_input_args = (inp.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(0), output[1].squeeze(0))


def generate_regression_criterion_inputs(make_input):
    # [FUNCTION DOC] HELPER: Builds three ModuleInput samples (reduction none, mean and sum) for
    # loss modules, each with 4-element unbatched input and target made by the supplied tensor
    # factory and checked with the criterion-aware no-batch-dim reference. Returns a list.
    return [
        ModuleInput(
            constructor_input=FunctionInput(reduction=reduction),
            forward_input=FunctionInput(make_input((4, )), make_input(4,)),
            reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True),
            desc=f'no_batch_dim_{reduction}'
        ) for reduction in ['none', 'mean', 'sum']]


def module_inputs_torch_nn_AvgPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AvgPool1d: unbatched input with kernel size 2
    # (checked with the no-batch-dim reference), a batched input, a stride variant and a stride
    # plus padding variant.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(kernel_size=2),
                    forward_input=FunctionInput(make_input((3, 6))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput(2),
                    forward_input=FunctionInput(make_input((2, 3, 6)))),
        ModuleInput(constructor_input=FunctionInput((2,), (2,)),
                    forward_input=FunctionInput(make_input((2, 3, 6))),
                    desc='stride'),
        ModuleInput(constructor_input=FunctionInput(2, 2, 1),
                    forward_input=FunctionInput(make_input((2, 3, 6))),
                    desc='stride_pad')]


def module_inputs_torch_nn_AvgPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AvgPool2d: unbatched and batched inputs, stride and
    # stride plus padding variants, and the same three variants again with divisor_override set.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput((2, 2)),
                    forward_input=FunctionInput(make_input((3, 6, 6))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput((2, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6)))),
        ModuleInput(constructor_input=FunctionInput((2, 2), (2, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='stride'),
        ModuleInput(constructor_input=FunctionInput((2, 2), (2, 2), (1, 1)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='stride_pad'),
        ModuleInput(constructor_input=FunctionInput((2, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='divisor'),
        ModuleInput(constructor_input=FunctionInput((2, 2), (2, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='divisor_stride'),
        ModuleInput(constructor_input=FunctionInput((2, 2), (2, 2), (1, 1), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='divisor_stride_pad')]



def module_inputs_torch_nn_AvgPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AvgPool3d: unbatched and batched inputs plus many
    # stride, padding and divisor_override combinations; several names mirror special GPU kernel
    # paths (fixed kernel width, general output, non-overlapping windows).
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput((2, 2, 2)),
                    forward_input=FunctionInput(make_input((3, 4, 4, 4))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput((2, 2, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4)))),
        ModuleInput(constructor_input=FunctionInput(2, (2, 2, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='stride'),
        ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='stride_pad'),
        ModuleInput(constructor_input=FunctionInput(4, 2, (1, 2, 1)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='stride_pad_gpu_fixedkw_output'),
        ModuleInput(constructor_input=FunctionInput((2, 4, 8), 1, (1, 1, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 2, 4, 8))),
                    desc='stride_pad_gpu_general_output'),
        ModuleInput(constructor_input=FunctionInput(3, 1, 0),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='stride1_pad0_gpu_input'),
        ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1)),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='stride_pad_gpu_input_nooverlap'),
        ModuleInput(constructor_input=FunctionInput((2, 2, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='divisor'),
        ModuleInput(constructor_input=FunctionInput(2, (2, 2, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='divisor_stride'),
        ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='divisor_stride_pad'),
        ModuleInput(constructor_input=FunctionInput(4, 2, (1, 2, 1), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='divisor_stride_pad_gpu_fixedkw_output'),
        ModuleInput(constructor_input=FunctionInput((2, 4, 8), 1, (1, 1, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 2, 4, 8))),
                    desc='divisor_stride_pad_gpu_general_output'),
        ModuleInput(constructor_input=FunctionInput(3, 1, 0, divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='divisor_stride1_pad0_gpu_input'),
        ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='divisor_stride_pad_gpu_input_nooverlap')]



def module_inputs_torch_nn_AdaptiveAvgPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AdaptiveAvgPool1d: output size 3 on a batched input,
    # the same on an unbatched input (no-batch-dim reference) and output size 1.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input((1, 3, 5))),
                    desc='one_output')]


def module_inputs_torch_nn_AdaptiveAvgPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AdaptiveAvgPool2d: single-int size, unbatched input,
    # 1x1 output, tuple size, and a tuple with a None entry meaning keep the input size.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 6))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='single_1x1output'),
        ModuleInput(constructor_input=FunctionInput((3, 4)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple'),
        ModuleInput(constructor_input=FunctionInput((3, None)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple_none')]

def module_inputs_torch_nn_AdaptiveAvgPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AdaptiveAvgPool3d: single-int size, unbatched input,
    # tuple size, tuple with a None entry, and a small input where the last dimension is shrunk.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 5, 2, 7))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 2, 7))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 3, 7))),
                    desc='tuple'),
        ModuleInput(constructor_input=FunctionInput((None, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 3, 7))),
                    desc='tuple_none'),
        ModuleInput(constructor_input=FunctionInput((3, 2, 2)),
                    forward_input=FunctionInput(make_input((1, 1, 3, 2, 6))),
                    desc='last_dim')]


def module_inputs_torch_nn_AdaptiveMaxPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AdaptiveMaxPool1d: output size 3 on batched and
    # on unbatched inputs.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_AdaptiveMaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AdaptiveMaxPool2d: single-int size, unbatched input,
    # tuple size, and a tuple with a None entry.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 6))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput((3, 4)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple'),
        ModuleInput(constructor_input=FunctionInput((3, None)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple_none')]


def module_inputs_torch_nn_AdaptiveMaxPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.AdaptiveMaxPool3d: single-int size, unbatched
    # input, tuple sizes (one with None), and two cases whose input sizes are not multiples of
    # the output sizes.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 6, 7))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='tuple'),
        ModuleInput(constructor_input=FunctionInput((3, None, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='tuple_none'),
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 12, 9, 3))),
                    desc='single_nonatomic'),
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 4, 10))),
                    desc='tuple_nonatomic')]


def module_inputs_torch_nn_BatchNorm1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.BatchNorm1d: affine default on 2D input, 3D input,
    # simple averaging (momentum None), no affine, no running statistics, 3D without affine,
    # zero-size batch, and no bias.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(10,),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='affine'),
        ModuleInput(constructor_input=FunctionInput(5,),
                    forward_input=FunctionInput(make_input((4, 5, 3))),
                    desc='3d_input'),
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, None),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='affine_simple_average'),
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='not_affine'),
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, 0.3, True, False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='not_tracking_stats'),
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((4, 5, 3))),
                    desc='3d_input_not_affine'),
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((0, 5, 9))),
                    desc='zero_batch'),
        ModuleInput(constructor_input=FunctionInput(10, bias=False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='affine_not_bias'),]


def module_inputs_torch_nn_BatchNorm2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.BatchNorm2d: default, simple averaging, custom
    # momentum, no affine, no running statistics, zero-size batch, and no bias on 4D inputs.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6)))),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, None),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='2d_simple_average'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.8),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='momentum'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.8, False),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='not_affine'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.8, True, False),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='not_tracking_stats'),
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((0, 5, 2, 2))),
                    desc='zero_batch'),
        ModuleInput(constructor_input=FunctionInput(3, bias=False),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='affine_not_bias'),]


def module_inputs_torch_nn_BatchNorm3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.BatchNorm3d: default, simple averaging, custom
    # momentum, no affine, no running statistics, zero-size batch, and no bias on 5D inputs.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4)))),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, None),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='3d_simple_average'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='momentum'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7, False),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='not_affine'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7, True, False),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='not_tracking_stats'),
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((0, 5, 2, 2, 2))),
                    desc='zero_batch'),
        ModuleInput(constructor_input=FunctionInput(3, bias=False),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='affine_not_bias'),]


def module_error_inputs_torch_nn_BatchNorm1d_2d_3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for BatchNorm1d/2d/3d: two samples with epsilon of -1
    # and 0 (input shape chosen from the module class under test) expecting a ValueError on
    # forward saying eps must be positive.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    if module_info.module_cls == torch.nn.BatchNorm1d:
        input_shape = (2, 10)
    elif module_info.module_cls == torch.nn.BatchNorm2d:
        input_shape = (2, 10, 5, 5)
    else:
        input_shape = (2, 10, 4, 4, 4)

    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, eps=-1.0),
                forward_input=FunctionInput(make_input(input_shape)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="eps must be positive"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, eps=0.0),
                forward_input=FunctionInput(make_input(input_shape)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="eps must be positive"
        ),
    ]


def module_inputs_torch_nn_ConvNd(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for the ConvNd family: requires the keyword N (number of
    # spatial dimensions) and reads optional lazy (constructor omits the input channel count)
    # and transposed (skips same-padding) flags. For each padding variant it builds a batched
    # sample and an unbatched sample (no-batch-dim reference) with kernel size 3, 4 input and
    # 5 output channels.
    # Raises KeyError if N is not supplied.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    N = kwargs['N']
    lazy = kwargs.get('lazy', False)
    transposed = kwargs.get('transposed', False)
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    conv_kwargs_list = [{}] if transposed else [{}, {'padding': 'same'}]
    kernel_size, C_in, C_out = 3, 4, 5
    input_no_batch_shape = (C_in,) + tuple(i + 3 for i in range(N))
    input_batch_shape = (2,) + input_no_batch_shape
    return [
        ModuleInput(constructor_input=(FunctionInput(C_out, kernel_size, **conv_kwargs) if lazy else
                                       FunctionInput(C_in, C_out, kernel_size, **conv_kwargs)),
                    forward_input=FunctionInput(make_input(
                        input_batch_shape if with_batch else input_no_batch_shape)),
                    desc=('' if with_batch else 'no_batch_dim'),
                    reference_fn=(None if with_batch else no_batch_dim_reference_fn))
        for with_batch, conv_kwargs in itertools.product([True, False], conv_kwargs_list)
    ]


def module_inputs_torch_nn_CosineEmbeddingLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.CosineEmbeddingLoss: cases for default, sum, mean and
    # none reduction and a margin of 0.7; two 15x10 inputs and a target of plus/minus one derived
    # from the sign of random values; the reference is cosineembeddingloss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('margin', {'margin': 0.7})
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i1, i2, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters
            # and returns the result of cosineembeddingloss_reference for the given input(s)
            # and the bound case options. The case options are bound as defaults so each sample
            # keeps its own.
            return cosineembeddingloss_reference(i1, i2, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 10)), make_input((15, 10)),
                                                    make_target((15,)).sign()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_ELU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ELU: alpha 2 with an explicit reference on a 3D
    # input, a scalar input, an unbatched input and a 4D input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2 * (i.exp() - 1))),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((2, 3, 2, 5))),
                    desc='4d_input')]


def module_inputs_torch_nn_CELU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.CELU: alpha 2 with an explicit reference on a 3D
    # input and a scalar input, plus an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2. * ((.5 * i).exp() - 1))),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2. * ((.5 * i).exp() - 1)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_GLU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.GLU: default dim, an explicit dim of 1 on a 3D input,
    # and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((5, 6)))),
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((5, 6, 7))),
                    desc='dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((4,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_GELU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.GELU: exact (none) approximation with an erf-based
    # reference on scalar and 3D inputs, and an unbatched input with default settings.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput('none'),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, x, *_: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput('none'),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, x, *_: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_ReLU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ReLU: scalar, unbatched, and 4D / 5D inputs labelled
    # as channels-last memory format samples.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='channels_last_mem_format'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
                    desc='channels_last_3d_mem_format')]


def module_inputs_torch_nn_ReLU6(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ReLU6: scalar, unbatched, and 4D / 5D inputs labelled
    # as channels-last memory format samples.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='channels_last_mem_format'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
                    desc='channels_last_3d_mem_format')]


def module_inputs_torch_nn_LeakyReLU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LeakyReLU: default slope, unbatched, a custom
    # negative slope, a zero slope and a scalar with custom slope.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(0.5),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    desc='with_negval'),
        ModuleInput(constructor_input=FunctionInput(0.0),
                    forward_input=FunctionInput(make_input((10, 10))),
                    desc='with_zero_negval'),
        ModuleInput(constructor_input=FunctionInput(0.5),
                    forward_input=FunctionInput(make_input(())),
                    desc='with_negval_scalar')]


def module_inputs_torch_nn_PReLU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.PReLU: scalar, unbatched, and 1d/2d/3d inputs each
    # with a single shared parameter and with one parameter per channel; the 1d/2d/3d references
    # use a clamp-based formula with the first parameter.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='1d'),
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 4))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='1d_multiparam'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='2d'),
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='2d_multiparam'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5, 6))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='3d'),
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5, 6))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='3d_multiparam')]


def module_inputs_torch_nn_SELU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.SELU: 3D input, unbatched input and scalar input
    # with default settings.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar')]


def module_inputs_torch_nn_SiLU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.SiLU: scalar and 3D inputs with a sigmoid-times-input
    # reference, and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, x, *_: x * torch.sigmoid(x),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((5, 6, 7))),
                    reference_fn=lambda m, p, x, *_: x * torch.sigmoid(x))]


def module_inputs_torch_nn_Softmax(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Softmax: dim 1 on a 10x20 input and dim 0 on a
    # scalar, both with explicit references, and an unbatched input with dim -1.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(1, True).expand(10, 20))),
        ModuleInput(constructor_input=FunctionInput(0),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(0, True)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(-1),
                    forward_input=FunctionInput(make_input((4, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softmax2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Softmax2d: a 4D input with an explicit reference and
    # an unbatched 3D input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((1, 3, 10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(1, False))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 4, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_LogSoftmax(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LogSoftmax: dim 1 on 2D and 4D inputs and dim 0 on a
    # scalar, all with explicit log-of-softmax references, and an unbatched input with dim -1.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(1, True).expand(10, 20)).log_()),
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((1, 3, 10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(1, False)).log_(),
                    desc='multiparam'),
        ModuleInput(constructor_input=FunctionInput(0),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(0, False)).log_(),
                    desc='multiparam_scalar'),
        ModuleInput(constructor_input=FunctionInput(-1),
                    forward_input=FunctionInput(make_input((4, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softmin(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Softmin: dim 1 on 2D and 4D inputs, dim 0 on a
    # scalar, and an unbatched input with dim -1.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((10, 20)))),
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((2, 3, 5, 10))),
                    desc='multidim'),
        ModuleInput(constructor_input=FunctionInput(0),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(-1),
                    forward_input=FunctionInput(make_input((3, 4, 10))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softplus(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Softplus: default, custom beta, beta with threshold,
    # beta with threshold on a scalar (explicit references), and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: torch.log1p(torch.exp(i))),
        ModuleInput(constructor_input=FunctionInput(2),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: 1. / 2. * torch.log1p(torch.exp(2 * i)),
                    desc='beta'),
        ModuleInput(constructor_input=FunctionInput(2, -100),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=(
                        lambda m, p, i: ((i * 2) > -100).type_as(i) * i
                        + ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log1p(torch.exp(2 * i))),
                    desc='beta_threshold'),
        ModuleInput(constructor_input=FunctionInput(2, -100),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=(
                        lambda m, p, i: ((i * 2) > -100).type_as(i) * i
                        + ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log1p(torch.exp(2 * i))),
                    desc='beta_threshold_scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Softshrink: default lambda, lambda 1 on 3D and scalar
    # inputs, and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5)))),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    desc='lambda'),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input(())),
                    desc='lambda_scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softsign(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Softsign: 3D and scalar inputs with an explicit
    # reference, and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: i.div(1 + torch.abs(i))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: i.div(1 + torch.abs(i)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Tanh(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Tanh: 4D, scalar and unbatched inputs.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]



def module_inputs_torch_nn_Tanhshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Tanhshrink: 4D, scalar and unbatched inputs.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Threshold(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Threshold: threshold 2 with replacement values 1 and
    # 10 on 4D input, a scalar input, and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='threshold_value'),
        ModuleInput(constructor_input=FunctionInput(2., 10.),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='large_value'),
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input(())),
                    desc='threshold_value_scalar'),
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Mish(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Mish: 3D and scalar inputs with an explicit reference
    # built from the softplus function, and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((5, 6, 7))),
                    reference_fn=lambda m, p, i: i * torch.tanh(F.softplus(i))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: i * torch.tanh(F.softplus(i)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_L1Loss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.L1Loss: a 2x3x4 input with an explicit per-batch
    # reference, a scalar input, plus the shared regression-criterion samples for each reduction.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4)),
                                                make_input((2, 3, 4))),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * sum((a - b).abs().sum()
                                                                         for a, b in zip(i, t, strict=True))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(()), make_input(())),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * (i - t).abs().sum(),
                    desc='scalar')] + generate_regression_criterion_inputs(make_input)


def module_inputs_torch_nn_SmoothL1Loss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.SmoothL1Loss: cases for default, sum, mean and none
    # reduction, each on a 5x10 input and on a scalar; the reference is smoothl1loss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)


    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of smoothl1loss_reference for the given input(s) and the bound
            # case options. The case options are bound as defaults so each sample keeps its own.
            return smoothl1loss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_input((5, 10))),
                        desc=desc,
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(()),
                                                    make_input(())),
                        desc=f'scalar_{desc}',
                        reference_fn=reference_fn)
        )

    return module_inputs



def module_inputs_torch_nn_BCELoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.BCELoss: cases for default, sum, mean, none
    # reduction and per-element weights on 15x10 inputs clamped to the open unit interval with
    # binary targets, plus a scalar-weight sample on scalar input; a nested reference computes
    # the expected loss.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('weights', {'weight': make_weight((10,))}),
    ]

    def bce_loss_reference_fn(m, p, i, t, reduction='mean', weight=None):
        # [FUNCTION DOC] HELPER: Expected binary cross entropy from probabilities: elementwise
        # loss, scaled by the weight if given, then reduced by none, mean (sum over element
        # count) or sum.
        result = -(t * i.log() + (1 - t) * (1 - i).log())

        if weight is not None:
            result = result * weight

        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()
        else:
            return result.sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 10), low=1e-2, high=1 - 1e-2),
                                                    make_target((15, 10)).gt(0).to(dtype)),
                        desc=desc,
                        reference_fn=partial(bce_loss_reference_fn, **constructor_kwargs))
        )

    scalar_weight = make_weight(())
    module_inputs.append(
        ModuleInput(constructor_input=FunctionInput(weight=scalar_weight),
                    forward_input=FunctionInput(make_input((), low=1e-2, high=1 - 1e-2),
                                                make_target(()).gt(0).to(dtype)),
                    desc='scalar_weight',
                    reference_fn=partial(bce_loss_reference_fn, weight=scalar_weight))
    )

    return module_inputs


def module_inputs_torch_nn_BCEWithLogitsLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.BCEWithLogitsLoss: cases for default, sum, mean, none
    # reduction, weights and scalar weights on 15x10 inputs with binary targets; a nested
    # reference computes the numerically stable loss.
    # KNOWN ISSUE: inputs are drawn from the narrow unit interval like the plain BCE
    # samples, which does not exercise large-magnitude logits; pos_weight is untested (a
    # TODO in the reference).
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('weights', {'weight': make_weight((10,))}),
        ('scalar_weights', {'weight': make_weight(())})
    ]

    def bce_withlogitsloss_reference_fn(m, p, i, t, reduction='mean', weight=None):
        # [FUNCTION DOC] HELPER: Expected binary cross entropy from logits using the
        # max-subtraction stable formulation, scaled by the weight if given, then reduced by none,
        # mean or sum. pos_weight is not supported (TODO in code).
        # TODO: add pos_weight to the definition here and corresponding SampleInputs
        max_val = (-i).clamp(min=0)
        result = (1 - t).mul_(i).add_(max_val).add_((-max_val).exp_().add_((-i - max_val).exp_()).log_())

        if weight is not None:
            result = result * weight

        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()
        else:
            return result.sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 10), low=1e-2, high=1 - 1e-2),
                                                    make_target((15, 10)).gt(0).to(dtype)),
                        desc=desc,
                        reference_fn=partial(bce_withlogitsloss_reference_fn, **constructor_kwargs))
        )

    return module_inputs


def module_inputs_torch_nn_CrossEntropyLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.CrossEntropyLoss: loops over the reductions and cases
    # (default, class weights, ignore_index, label smoothing, both): each builds class-index
    # targets on 4D, 3D, 2D and higher-dimensional inputs; when ignore_index is absent it also
    # builds class-probability targets on four shapes and an unbatched sample with the criterion
    # reference. The reference is cross_entropy_loss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    reductions: list[str] = ['mean', 'sum', 'none']
    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('weights', {'weight': make_weight((3,))}),
        ('ignore_index', {'ignore_index': 1}),
        ('label_smoothing', {'label_smoothing': 0.15}),
        ('ignore_index_label_smoothing', {'ignore_index': 1, 'label_smoothing': 0.15})
    ]

    module_inputs = []
    for reduction, (desc, constructor_kwargs) in product(reductions, cases):
        def reference_fn(m, p, i, t, reduction=reduction, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of cross_entropy_loss_reference for the given input(s) and the
            # bound case options. The reduction and case options are bound as defaults so each
            # sample keeps its own.
            return cross_entropy_loss_reference(i, t, reduction=reduction, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 5, 5)),
                                                    make_target((2, 5, 5), low=0, high=3)),
                        desc=f"4d_{desc}_{reduction}",
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 5)),
                                                    make_target((2, 5), low=0, high=3)),
                        desc=f"3d_{desc}_{reduction}",
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3)),
                                                    make_target((2), low=0, high=3)),
                        desc=f"2d_{desc}_{reduction}",
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 5, 5, 2, 2)),
                                                    make_target((2, 5, 5, 2, 2), low=0, high=3)),
                        desc=f"higher_dim_{desc}_{reduction}",
                        reference_fn=reference_fn)
        )

        if constructor_kwargs.get('ignore_index', None) is None:
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((5, 3, 4, 2)),
                                                        make_input((5, 3, 4, 2)).softmax(dim=1)),
                            desc=f"4d_prob_target_{desc}_{reduction}",
                            reference_fn=reference_fn)
            )
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((5, 3, 4)),
                                                        make_input((5, 3, 4)).softmax(dim=1)),
                            desc=f"3d_prob_target_{desc}_{reduction}",
                            reference_fn=reference_fn)
            )
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((5, 3)),
                                                        make_input((5, 3)).softmax(dim=1)),
                            desc=f"2d_prob_target_{desc}_{reduction}",
                            reference_fn=reference_fn)
            )
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((2, 3, 5, 5, 2, 2)),
                                                        make_input((2, 3, 5, 5, 2, 2)).softmax(dim=1)),
                            desc=f"higher_dim_prob_target_{desc}_{reduction}",
                            reference_fn=reference_fn)
            )
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((3,)),
                                                        make_target((), low=0, high=3)),
                            desc=f"no_batch_dim_{desc}_{reduction}",
                            reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True))
            )

    return module_inputs


def module_error_inputs_torch_nn_CrossEntropyLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.CrossEntropyLoss: seven forward-stage errors:
    # integer target with probability shape, non-long target with index shape, batch size
    # mismatch, wrongly shaped weight, ignore_index with probability targets, label smoothing out
    # of range, and mismatched target spatial size.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    return [
        # Non-floating-point target with same shape as input (soft-target path)
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(),
                forward_input=FunctionInput(
                    make_input((2, 3)),
                    torch.zeros((2, 3), device=device, dtype=torch.long),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="Expected floating point type for target with class probabilities",
        ),
        # Non-long target with different shape from input (class-index path)
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(),
                forward_input=FunctionInput(
                    make_input((2, 3)),
                    make_input((2,)),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="expected target dtype to be Long or Byte",
        ),
        # Batch size mismatch
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(),
                forward_input=FunctionInput(
                    make_input((2, 3)),
                    torch.tensor([0, 1, 2], device=device, dtype=torch.long),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="Expected input batch_size .* to match target batch_size",
        ),
        # Weight wrong shape
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(weight=make_weight((2,))),
                forward_input=FunctionInput(
                    make_input((2, 3)),
                    torch.tensor([0, 1], device=device, dtype=torch.long),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="weight tensor should be defined either for all .* classes or no classes",
        ),
        # Prob target with ignore_index
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(ignore_index=1),
                forward_input=FunctionInput(
                    make_input((2, 3)),
                    make_input((2, 3)).softmax(dim=1).detach(),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="ignore_index is not supported for floating point target",
        ),
        # label_smoothing out of range
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(label_smoothing=1.5),
                forward_input=FunctionInput(
                    make_input((2, 3)),
                    torch.tensor([0, 1], device=device, dtype=torch.long),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="label_smoothing must be between 0.0 and 1.0",
        ),
        # Target spatial size mismatch (3D input)
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(),
                forward_input=FunctionInput(
                    make_input((2, 3, 5)),
                    torch.zeros((2, 4), device=device, dtype=torch.long),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="Expected target size",
        ),
    ]


def module_inputs_torch_nn_LinearCrossEntropyLoss(module_info, device, dtype, requires_grad, training, **kwargs_):
    # [FUNCTION DOC] Sample generator for nn.LinearCrossEntropyLoss: builds a large grid of
    # samples for the fused linear plus cross-entropy module. Reads optional keywords
    # allow_retain_graph (default true) and acc_dtype. Samples sweep reductions, ignore_index,
    # label smoothing, class weights, extra output dims, index and probability targets and
    # chunking options, plus bias coverage and chunked probability targets. The reference reshapes
    # the first parameter and optional bias and calls linear_cross_entropy_loss_reference.
    # The count and order of samples must stay identical across devices because tests depend on
    # it, and the sample list is append-only (entries consume fixed random draws).
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    # Return a list of module_inputs.
    #
    # Important: module_inputs size and the ordering of its items must
    # be the same for all devices.  There exists tests that
    # correctness depend on this requirement.
    allow_retain_graph = kwargs_.get('allow_retain_graph', True)
    acc_dtype = kwargs_.get('acc_dtype')

    def make_input(batch_dims, in_features):
        # [FUNCTION DOC] HELPER: Returns a standard-normal tensor with the batch dimensions
        # followed by the feature count, using the enclosing device, dtype and requires_grad.
        return torch.randn((*batch_dims, in_features), device=device, dtype=dtype, requires_grad=requires_grad)

    def reference_fn(m, p, i, t):
        # [FUNCTION DOC] HELPER: Sample reference: reshapes the module's first parameter to
        # classes, output dims and in_features (and the optional second parameter as bias) and
        # returns linear_cross_entropy_loss_reference using the module's weight, reduction,
        # ignore_index and label_smoothing.
        # p[0] is linear.weight; p[1] is linear.bias when bias=True.
        linear_weight = p[0].reshape(m.num_classes, *m.out_features, i.shape[-1])
        linear_bias = (
            p[1].reshape(m.num_classes, *m.out_features) if len(p) > 1 else None
        )
        return linear_cross_entropy_loss_reference(
            i, linear_weight, t,
            linear_bias=linear_bias,
            weight=m.weight,
            reduction=m.reduction, ignore_index=m.ignore_index, label_smoothing=m.label_smoothing)

    def make_target(num_classes, shape, target_dtype):
        # [FUNCTION DOC] HELPER: Creates a target: for a floating dtype, uniform values in the
        # unit interval of the given shape (probability target); otherwise random integer class
        # indices below num_classes. Never requires grad.
        if target_dtype.is_floating_point:
            return make_tensor(shape, low=0, high=1, device=device, dtype=target_dtype, requires_grad=False)
        else:
            return torch.randint(
                0,
                num_classes,
                shape,
                device=device,
                dtype=target_dtype,
                requires_grad=False,
            )

    def sizes_and_options():
        # [FUNCTION DOC] HELPER: Generator yielding pairs of (batch count, input features, class
        # count) and chunking options for two size configurations (batched and unbatched). Yields
        # no-options first; if acc_dtype was requested it yields only an accumulator dtype option;
        # otherwise an empty options set and, for batched sizes, fixed chunk size 2 and
        # aspect-ratio chunking 2 and 4.
        # The order is append-only because each entry consumes fixed random draws.
        # APPEND-ONLY: never reorder/insert/remove entries -- each consumes
        # fixed RNG draws, so a mid-stream change shifts later samples and
        # moves the cancellation-sensitive index-target ULP caps in test_nn.py.
        for sizes in [(8, 5, 4), (None, 8, 4)]:
            yield sizes, None
            num_batches, in_features, num_classes = sizes
            if acc_dtype is not None:
                yield sizes, dict(acc_dtype=acc_dtype, chunking_method="aspect_ratio")
                continue
            # unspecified options use the auto resolution (aspect_ratio
            # factor 1, capped to stay at or below the reference peak):
            yield sizes, dict()

            if num_batches is not None:
                # fixed chunk size reduces memory usage but may reduce
                # processing performance:
                yield sizes, dict(batch_chunk_size=2)
                # alternatively, chunk sizes via an explicit chunking method:
                yield sizes, dict(chunking_method="aspect_ratio:2")
                yield sizes, dict(chunking_method="aspect_ratio:4")

    def samples():
        # [FUNCTION DOC] HELPER: Generator yielding (module args, module kwargs, (input, target))
        # triples. First a product over reduction, ignore_index, label smoothing, class weight and
        # extra output dims (skipping impossible or redundant combinations) for index and
        # probability targets, with extra samples exercising out-of-range ignore_index and
        # all-ignored targets; then minimal bias coverage (with and without chunked options);
        # finally chunked probability-target samples for mean and sum.
        # Targets are adjusted so mean reduction never normalizes by zero valid entries.
        for (num_batches, in_features, num_classes), options in sizes_and_options():
            if num_batches is None:
                batch_dims = ()
            else:
                batch_dims = (num_batches,)
            weights = [None, torch.exp(torch.randn(num_classes, device=device, dtype=dtype, requires_grad=False))]

            # generate samples for LinearCrossEntropyLoss and its forward:
            for reduction, ii, ls, w, of in product(
                    ["sum", "mean", "none"],
                    [None, -100, (num_classes - 1) // 2, num_classes + 5],
                    [0.0, 0.1],
                    weights,
                    [(), (3, 2, 4)]):
                if num_batches is None and of:
                    # K-dimensional loss requires batches dimension
                    continue
                if ii is not None and reduction == "sum":
                    # skip some samples, just to reduce the size of samples set
                    continue
                module_args = (in_features, num_classes)
                module_kwargs = dict(
                    out_features=of,
                    device=device,
                    dtype=dtype,
                    reduction=reduction,
                    weight=w,
                    ignore_index=ii,
                    label_smoothing=ls,
                    options=torch.nn.LinearCrossEntropyOptions(allow_retain_graph=allow_retain_graph, **options) if options is not None else None
                )
                for target_dtype in [torch.int64, dtype]:
                    if target_dtype.is_floating_point:
                        target_shape = (*batch_dims, num_classes, *of)
                        if ii is not None:
                            # ignore_index is not supported for floating point target
                            continue
                        if options is not None:
                            # chunked probability-target samples are
                            # appended after the main loop (so the RNG
                            # draw order of these samples is unchanged)
                            continue
                    else:
                        target_shape = (*batch_dims, *of)
                    input = make_input(batch_dims, in_features)
                    target = make_target(num_classes, target_shape, target_dtype)
                    if (
                            target.device.type != "meta"
                            and not target_dtype.is_floating_point and reduction == "mean"
                            and ii is not None and ii >= 0 and ii < num_classes and torch.all(target == ii)
                    ):
                        # ensures valid normalization:
                        target[0 if target.shape else ()] = (ii + 1) % num_classes

                    yield module_args, module_kwargs, (input, target)

                    if (
                            not target_dtype.is_floating_point
                            and target_shape
                            and num_batches > 1
                            and ii is not None
                    ):
                        # target may contain out-of-range ii values
                        input = make_input(batch_dims, in_features)
                        target = make_target(num_classes, target_shape, target_dtype)
                        target[num_batches // 2] = ii
                        if ii < 0 or ii >= num_classes:
                            # tests the correctness of out-of-range ii
                            # mapping to 0 (see the batch chunking
                            # logic)
                            target[0] = 0
                        yield module_args, module_kwargs, (input, target)

                    if (
                            not target_dtype.is_floating_point and reduction != "mean" and ii is not None
                    ):
                        # target is completely filled with ii
                        input = make_input(batch_dims, in_features)
                        target = torch.full_like(target, ii)
                        yield module_args, module_kwargs, (input, target)

        # Minimal bias=True coverage: one sample per (size, out_features,
        # options-variant) with the rest of the parametrization fixed at
        # defaults. Bias is orthogonal to reduction / ignore_index /
        # label_smoothing / weight / target_dtype (linear adds a
        # constant to every logit element regardless of how
        # cross_entropy reduces or weights them), so we don't expand
        # against those axes. We DO emit one chunked variant
        # (``options != None``) per (size, out_features) so the chunked
        # ``linear_bias`` path -- including the mixed-precision
        # acc_dtype scratch + post-yield commit on fp16/bf16 inputs --
        # gets gradient coverage through ``_test_linear_cross_entropy_loss``.
        for sizes in [(8, 5, 4), (None, 8, 4)]:
            num_batches, in_features, num_classes = sizes
            batch_dims = () if num_batches is None else (num_batches,)
            for of in [(), (3, 2, 4)]:
                if num_batches is None and of:
                    continue  # K-dim loss requires a batch dim.
                module_args = (in_features, num_classes)
                base_module_kwargs = dict(
                    out_features=of,
                    bias=True,
                    device=device,
                    dtype=dtype,
                    reduction="mean",
                    weight=None,
                    ignore_index=None,
                    label_smoothing=0.0,
                )
                # options=None: reference-path bias coverage.
                # options=LinearCrossEntropyOptions(...): chunked-path
                #   bias coverage (only fires when out_features == ();
                #   K-dim chunked + bias falls back to the reference
                #   path with a warning -- not a useful test sample).
                #   ``batch_chunk_size=2`` forces >=2 chunks on every
                #   device so the post-yield ``bias_grad_acc.zero_()``
                #   reset between chunks gets exercised even on CPU
                #   (where the ``aspect_ratio`` auto-heuristic for the
                #   small (8, 5, 4) shape would otherwise produce a
                #   single chunk).
                options_variants = [None]
                if not of:
                    options_variants.append(
                        torch.nn.LinearCrossEntropyOptions(
                            allow_retain_graph=allow_retain_graph,
                            batch_chunk_size=2,
                        )
                    )
                for options in options_variants:
                    module_kwargs = dict(base_module_kwargs, options=options)
                    input = make_input(batch_dims, in_features)
                    target = make_target(num_classes, (*batch_dims, *of), torch.int64)
                    yield module_args, module_kwargs, (input, target)

        # Chunked probability-target coverage (mean/sum; reduction='none'
        # with a probability target falls back to the reference).
        # Appended after all other samples so the RNG draw order -- and
        # therefore the data the calibrated index-target ULP caps were
        # measured on -- is unchanged.
        for sizes in [(8, 5, 4), (None, 8, 4)]:
            num_batches, in_features, num_classes = sizes
            batch_dims = () if num_batches is None else (num_batches,)
            weights = [None, torch.exp(torch.randn(num_classes, device=device, dtype=dtype, requires_grad=False))]
            for reduction, w in product(["mean", "sum"], weights):
                if acc_dtype is not None:
                    options = dict(acc_dtype=acc_dtype, chunking_method="aspect_ratio")
                elif num_batches is not None:
                    # batch_chunk_size=2 forces >=2 chunks on every device.
                    options = dict(batch_chunk_size=2)
                else:
                    options = dict()
                module_args = (in_features, num_classes)
                module_kwargs = dict(
                    out_features=(),
                    device=device,
                    dtype=dtype,
                    reduction=reduction,
                    weight=w,
                    ignore_index=None,
                    label_smoothing=0.0,
                    options=torch.nn.LinearCrossEntropyOptions(
                        allow_retain_graph=allow_retain_graph, **options
                    ),
                )
                input = make_input(batch_dims, in_features)
                target = make_target(num_classes, (*batch_dims, num_classes), dtype)
                yield module_args, module_kwargs, (input, target)

    module_inputs = []
    for module_args, module_kwargs, (input, target) in samples():
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(*module_args, **module_kwargs),
                forward_input=FunctionInput(input, target),
                reference_fn=reference_fn)
        )
    return module_inputs


def module_error_inputs_torch_nn_LinearCrossEntropyLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.LinearCrossEntropyLoss: five errors: integer
    # probability target, batch size mismatch, input feature mismatch, weight of wrong shape
    # (construction) and label smoothing out of range (construction).
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return [
        # Non-floating-point target with same shape as input (soft-target path)
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(3, 2, device=device, dtype=dtype),
                forward_input=FunctionInput(
                    make_input((4, 3)),
                    torch.zeros((4, 2), device=device, dtype=torch.long),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="Expected floating point type for target with class probabilities",
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(3, 2, device=device, dtype=dtype),
                forward_input=FunctionInput(
                    make_input((4, 3)),
                    torch.zeros((5, 2), device=device, dtype=dtype),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="Expected input batch_size [(]4[)] to match target batch_size [(]5[)]",
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(3, 2, device=device, dtype=dtype),
                forward_input=FunctionInput(
                    make_input((4, 5)),
                    torch.zeros((4, 2), device=device, dtype=dtype),
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="expected equal input and linear_weight last dimensions [(]in_features[)], got 5 and 3, respectively",
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(3, 2, weight=torch.ones(3, device=device, dtype=dtype),
                                                device=device, dtype=dtype),
                forward_input=FunctionInput(
                    make_input((4, 3)),
                    torch.zeros((4, 2), device=device, dtype=dtype),
                ),
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=RuntimeError,
            error_regex="expected weight shape to be [(]2,[)], got [(]3,[)]",
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(3, 2, device=device, dtype=dtype, label_smoothing=2.0),
                forward_input=FunctionInput(
                    make_input((4, 3)),
                    torch.zeros((4, 2), device=device, dtype=dtype),
                ),
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=RuntimeError,
            error_regex="expected label_smoothing to be in range [[]0, 1[]], got 2.0",
        ),
    ]


def module_inputs_torch_nn_CTCLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.CTCLoss: cases for default, sum, mean and none
    # reduction and blank index 14, for int and long targets; inputs are 50x3x15 log-probabilities
    # and each case uses padded 2D or concatenated 1D targets with lengths given as int tuples or
    # tensors (four samples per case and dtype). The reference is ctcloss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('blank', {'blank': 14})
    ]
    target_dtypes = [torch.int, torch.long]

    module_inputs = []
    for target_dtype, (desc, constructor_kwargs) in product(target_dtypes, cases):
        def reference_fn(m, p, i, t, il, tl, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of ctcloss_reference for the given input(s) and the bound case
            # options. The case options are bound as defaults so each sample keeps its own.
            return ctcloss_reference(i, t, il, tl, **constructor_kwargs)

        blank = constructor_kwargs.get('blank', 0)
        low = 0 if blank == 14 else 1
        high = 14 if blank == 14 else 15

        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((3, 30), dtype=target_dtype, low=low, high=high),
                                            (50, 50, 50), (30, 25, 20)),
                desc=f'{desc}_lengths_intlists',
                reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((3, 30), dtype=target_dtype, low=low, high=high),
                                            torch.tensor((50, 50, 50), device=device),
                                            torch.tensor((30, 25, 20), device=device)),
                desc=f'{desc}_lengths_tensors',
                reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((30 + 25 + 20,), dtype=target_dtype, low=low, high=high),
                                            (50, 50, 50), (30, 25, 20)),
                desc=f'{desc}_1d_target_lengths_intlists',
                reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((30 + 25 + 20,), dtype=target_dtype, low=low, high=high),
                                            torch.tensor((50, 50, 50), device=device),
                                            torch.tensor((30, 25, 20), device=device)),
                desc=f'{desc}_1d_target_lengths_tensors',
                reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_GroupNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.GroupNorm: ten samples on 1D and 2D inputs covering
    # affine, no bias, plain group norm, a large batch, and the instance-norm-like (groups equal
    # channels) and layer-norm-like (one group) configurations without affine.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(3, 6, 1e-3),
            forward_input=FunctionInput(make_input((4, 6, 5))),
            desc='1d_affine'),
        ModuleInput(
            constructor_input=FunctionInput(3, 6, 1e-3, bias=False),
            forward_input=FunctionInput(make_input((4, 6, 5))),
            desc='1d_affine_not_bias'),
        ModuleInput(
            constructor_input=FunctionInput(3, 12, 1e-3),
            forward_input=FunctionInput(make_input((4, 12))),
            desc='1d_affine_GN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 6, 1e-3),
            forward_input=FunctionInput(make_input((150, 6))),
            desc='1d_affine_large_batch'),
        ModuleInput(
            constructor_input=FunctionInput(5, 5, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_affine_IN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 10, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 10))),
            desc='1d_no_affine_LN'),
        ModuleInput(
            constructor_input=FunctionInput(3, 6, 1e-3),
            forward_input=FunctionInput(make_input((4, 6, 2, 3))),
            desc='2d_affine'),
        ModuleInput(
            constructor_input=FunctionInput(3, 9, 1e-3, bias=False),
            forward_input=FunctionInput(make_input((4, 9, 2, 3))),
            desc='2d_affine_not_bias'),
        ModuleInput(
            constructor_input=FunctionInput(3, 3, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 3, 2, 3))),
            desc='2d_no_affine_IN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 3, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 3, 2, 3))),
            desc='2d_no_affine_LN'),]


def module_error_inputs_torch_nn_GroupNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.GroupNorm: two construction-time
    # ValueError samples where the channel count is not divisible by the group count, with
    # the numbers in the message.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    """
    Error inputs for GroupNorm that test error messages include actual values.
    """
    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(3, 10),  # num_groups=3, num_channels=10
                forward_input=FunctionInput(),  # Not needed for construction error
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=ValueError,
            error_regex=r"num_channels \(10\) must be divisible by num_groups \(3\)"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(5, 13),  # num_groups=5, num_channels=13
                forward_input=FunctionInput(),  # Not needed for construction error
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=ValueError,
            error_regex=r"num_channels \(13\) must be divisible by num_groups \(5\)"
        ),
    ]


def module_inputs_torch_nn_Hardshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Hardshrink: lambda 2 on 4D and scalar inputs, and an
    # unbatched input with default settings.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(2.),
            forward_input=FunctionInput(make_input((4, 3, 2, 4))),
        ),
        ModuleInput(
            constructor_input=FunctionInput(2.),
            forward_input=FunctionInput(make_input(())),
            desc='scalar',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        )
    ]


def module_inputs_torch_nn_Hardswish(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Hardswish: an unbatched input and a 4D input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 2, 5))),
            desc='4d_input')
    ]


def module_inputs_torch_nn_Hardtanh(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Hardtanh: default clamp range with an explicit clamp
    # reference on 3D and scalar inputs, and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((3, 2, 5))),
            reference_fn=lambda m, p, i: i.clamp(-1, 1),
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(())),
            reference_fn=lambda m, p, i: i.clamp(-1, 1),
            desc='scalar',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        )
    ]


def module_inputs_torch_nn_HingeEmbeddingLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.HingeEmbeddingLoss: cases for default, sum, mean,
    # none reduction and a margin of 0.5, each on a 10-element input and on a scalar; targets are
    # plus/minus one; the reference is hingeembeddingloss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('margin', {'margin': 0.5})
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters
            # and returns the result of hingeembeddingloss_reference for the given input(s) and
            # the bound case options. The case options are bound as defaults so each sample
            # keeps its own.
            return hingeembeddingloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((10,)),
                                                    make_target((10,)).gt(0).to(dtype).mul_(2).sub_(1)),
                        desc=desc,
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(()),
                                                    make_target(()).gt(0).to(dtype).mul_(2).sub_(1)),
                        desc=f'scalar_{desc}',
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_HuberLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.HuberLoss: cases for default, sum, mean and none
    # reduction on 5x10 inputs; the reference is huberloss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of huberloss_reference for the given input(s) and the bound case
            # options. The case options are bound as defaults so each sample keeps its own.
            return huberloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_input((5, 10))),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_InstanceNormNd(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for the InstanceNormNd family: requires the keyword N (1, 2
    # or 3) and reads an optional lazy flag (constructor omits the feature count). Samples cover
    # default settings, tracked statistics without affine, unbatched variants with the
    # no-batch-dim reference, affine, and affine without bias.
    # Raises KeyError if N is not supplied.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    lazy = kwargs.get('lazy', False)
    N = kwargs['N']
    num_features, eps, momentum, affine, track_running_stats = 3, 1e-3, 0.3, False, True
    input_no_batch_shape_dict = {1: (3, 15), 2: (3, 6, 6), 3: (3, 4, 4, 4)}
    input_no_batch_shape = input_no_batch_shape_dict[N]
    input_batch_shape = (4,) + input_no_batch_shape

    return [
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum) if lazy else FunctionInput(num_features, eps, momentum)
            ),
            forward_input=FunctionInput(make_input(input_batch_shape))),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum, affine, track_running_stats) if lazy else
                FunctionInput(num_features, eps, momentum, affine, track_running_stats)
            ),
            forward_input=FunctionInput(make_input(input_batch_shape)),
            desc='tracking_stats'),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum) if lazy else FunctionInput(num_features, eps, momentum)
            ),
            forward_input=FunctionInput(make_input(input_no_batch_shape)),
            reference_fn=no_batch_dim_reference_fn,
            desc='tracking_stats_no_batch_dim'),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum, affine, track_running_stats) if lazy else
                FunctionInput(num_features, eps, momentum, affine, track_running_stats)
            ),
            forward_input=FunctionInput(make_input(input_no_batch_shape)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum, affine=True) if lazy else
                FunctionInput(num_features, eps, momentum, affine=True)
            ),
            forward_input=FunctionInput(make_input(input_batch_shape)),
            desc='affine'),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum, affine=True, bias=False) if lazy else
                FunctionInput(num_features, eps, momentum, affine=True, bias=False)
            ),
            forward_input=FunctionInput(make_input(input_batch_shape)),
            desc='affine_not_bias'),]

def module_inputs_torch_nn_LayerNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LayerNorm: seven samples covering 1D and 3D
    # normalized shapes with and without elementwise affine, a large batch, an empty batch, and
    # affine without bias.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((128, 5, 5))),
            desc='1d_elementwise_affine_large_batch'),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_no_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((0, 5))),
            desc='1d_empty_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, elementwise_affine=True, bias=False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine_no_bias'),
    ]

def module_inputs_torch_nn_RMSNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.RMSNorm: six samples (1D and 3D normalized shapes,
    # with and without elementwise affine, large batch, empty batch) each using a nested reference
    # that normalizes by the root mean square in float32.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def rms_norm_reference_fn(m, p, i):
        # [FUNCTION DOC] HELPER: Expected RMS normalization: casts the input to float32, divides
        # by the root of the mean of squares over the normalized trailing dims plus epsilon,
        # multiplies by the weight if present, and casts back to the input dtype.
        # KNOWN ISSUE: a fallback epsilon for a module whose eps is None is computed but the
        # formula uses the module's eps directly, so that case would fail with a type error.
        eps = m.eps
        if eps is None:
            eps = torch.finfo(i.dtype).eps
        ndim = i.ndim
        normalized_shape = m.normalized_shape
        weight = m.weight
        dims = [ndim - i - 1 for i in range(len(normalized_shape))]
        upcasted_i = i.float()
        result = upcasted_i * torch.rsqrt(upcasted_i.pow(2).mean(dim=dims, keepdim=True) + m.eps)
        if weight is not None:
            result *= weight
        return result.type_as(i)

    return [
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((128, 5, 5))),
            desc='1d_elementwise_affine_large_batch',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_no_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((0, 5))),
            desc='1d_empty_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
    ]


def module_inputs_torch_nn_LocalResponseNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LocalResponseNorm: 1D, 2D (uneven padding) and 3D
    # inputs with custom parameters.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(3,),
            forward_input=FunctionInput(make_input((1, 5, 7))),
            desc='1d'),
        ModuleInput(
            constructor_input=FunctionInput(2,),
            forward_input=FunctionInput(make_input((1, 5, 7, 7))),
            desc='2d_uneven_pad'),
        ModuleInput(
            constructor_input=FunctionInput(1, 1., 0.5, 2.),
            forward_input=FunctionInput(make_input((1, 5, 7, 7, 7))),
            desc='3d_custom_params'),
    ]


def module_inputs_torch_nn_LPPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LPPool1d: a norm of 1.5, an integer norm with kernel
    # and stride on batched input, and the same on an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1.5, 2),
            forward_input=FunctionInput(make_input((1, 3, 7))),
            desc='norm'),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 3),
            forward_input=FunctionInput(make_input((1, 3, 7)))),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 3),
            forward_input=FunctionInput(make_input((3, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),
    ]



def module_inputs_torch_nn_LPPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LPPool2d: integer norm with kernel on batched and
    # unbatched inputs, and a fractional norm of 1.5.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 2),
            forward_input=FunctionInput(make_input((1, 3, 7, 7)))),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 2),
            forward_input=FunctionInput(make_input((3, 7, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),
        ModuleInput(
            constructor_input=FunctionInput(1.5, 2),
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),
            desc='norm'),
    ]


def module_inputs_torch_nn_LPPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LPPool3d: integer norm with kernel on batched and
    # unbatched inputs, and a fractional norm of 1.5.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 2),
            forward_input=FunctionInput(make_input((1, 3, 7, 7, 7)))),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 2),
            forward_input=FunctionInput(make_input((3, 7, 7, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),
        ModuleInput(
            constructor_input=FunctionInput(1.5, 2),
            forward_input=FunctionInput(make_input((1, 3, 7, 7, 7))),
            desc='norm'),
    ]


def module_inputs_torch_nn_MaxPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MaxPool1d: kernel 4 on 3D input, kernel with stride,
    # and return_indices enabled.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(4),
            forward_input=FunctionInput(make_input((2, 10, 4))),
            desc='3d_input'),
        ModuleInput(
            constructor_input=FunctionInput(4, 4),
            forward_input=FunctionInput(make_input((2, 10, 4))),
            desc='stride'),
        ModuleInput(
            constructor_input=FunctionInput(4, return_indices=True),
            forward_input=FunctionInput(make_input((2, 10, 4))),
            desc='return_indices'),
    ]


def module_inputs_torch_nn_MaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MaxPool2d: 3x3 kernel with stride 2 and padding 1 on
    # 3D input, on 4D input, and with return_indices enabled.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1)),
            forward_input=FunctionInput(make_input((3, 7, 7))),
            desc='3d_input'),
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1)),
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),
            desc='4d_input'),
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1), return_indices=True),
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),
            desc='return_indices'),
    ]


def module_error_inputs_torch_nn_MaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.MaxPool2d: three forward-stage RuntimeError
    # samples: 2D input, 5D input, and padding larger than half the kernel size.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    """
    Error inputs for MaxPool2d that test error messages for invalid inputs.
    """
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        # Wrong input dimensions: 2D input instead of 3D/4D
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(2),
                forward_input=FunctionInput(make_input((3, 4))),  # 2D input
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex=r"non-empty 3D or 4D \(batch mode\) tensor expected for input"
        ),
        # Wrong input dimensions: 5D input
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(2),
                forward_input=FunctionInput(make_input((1, 2, 3, 4, 5))),  # 5D input
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex=r"non-empty 3D or 4D \(batch mode\) tensor expected for input"
        ),
        # Invalid padding: padding > kernel_size / 2
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(3, padding=5),  # kernel=3, pad=5 > 3/2
                forward_input=FunctionInput(make_input((1, 1, 10, 10))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex=r"pad should be at most half of effective kernel size"
        ),
    ]


def module_inputs_torch_nn_MaxPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MaxPool3d: default stride, explicit stride, stride
    # with padding, and return_indices enabled.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput((2, 2, 2)),
            forward_input=FunctionInput(make_input((2, 3, 5, 5, 5)))),
        ModuleInput(
            constructor_input=FunctionInput(2, (2, 2, 2)),
            forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
            desc='stride'),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, (1, 1, 1)),
            forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
            desc='stride_padding'),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, (1, 1, 1), return_indices=True),
            forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
            desc='return_indices'),
    ]


def module_inputs_torch_nn_FractionalMaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.FractionalMaxPool2d: ratio-based and size-based
    # output with explicit random samples, return_indices, and unbatched variants checked with the
    # no-batch-dim reference. The nested sample factory creates double-precision uniform values
    # regardless of dtype.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_random_samples():
        # [FUNCTION DOC] HELPER: Returns a fresh double-precision tensor of uniform random
        # numbers in the unit interval on the target device, used as the fractional
        # pooling window offsets.
        return torch.empty((1, 3, 2), dtype=torch.double, device=device).uniform_()

    return [
        ModuleInput(
            constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((1, 3, 5, 7))),
            desc='ratio'),
        ModuleInput(
            constructor_input=FunctionInput((2, 3), output_size=(4, 3), _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((1, 3, 7, 6))),
            desc='size'),
        ModuleInput(
            constructor_input=FunctionInput(
                2, output_ratio=0.5, _random_samples=make_random_samples(), return_indices=True
            ),
            forward_input=FunctionInput(make_input((1, 3, 5, 7))),
            desc='ratio_return_indices'),
        ModuleInput(
            constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((3, 5, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='ratio_no_batch_dim'),
        ModuleInput(
            constructor_input=FunctionInput((2, 3), output_size=(4, 3), _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((3, 7, 6))),
            reference_fn=no_batch_dim_reference_fn,
            desc='size_no_batch_dim'),
    ]


def module_inputs_torch_nn_FractionalMaxPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.FractionalMaxPool3d: ratio-based, size-based and
    # asymmetric-size output with explicit random samples, return_indices, and unbatched variants
    # checked with the no-batch-dim reference. The nested sample factory creates double-precision
    # uniform values regardless of dtype.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_random_samples():
        # [FUNCTION DOC] HELPER: Returns a fresh double-precision tensor of uniform random
        # numbers in the unit interval on the target device, used as the fractional
        # pooling window offsets.
        return torch.empty((2, 4, 3), dtype=torch.double, device=device).uniform_()

    return [
        ModuleInput(
            constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((2, 4, 5, 5, 5))),
            desc='ratio'),
        ModuleInput(
            constructor_input=FunctionInput((2, 2, 2), output_size=(4, 4, 4), _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((2, 4, 7, 7, 7))),
            desc='size'),
        ModuleInput(
            constructor_input=FunctionInput((4, 2, 3), output_size=(10, 3, 2), _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((2, 4, 16, 7, 5))),
            desc='asymsize'),
        ModuleInput(
            constructor_input=FunctionInput(
                2, output_ratio=0.5, _random_samples=make_random_samples(), return_indices=True
            ),
            forward_input=FunctionInput(make_input((2, 4, 5, 5, 5))),
            desc='ratio_return_indices'),
        ModuleInput(
            constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((4, 5, 5, 5))),
            reference_fn=no_batch_dim_reference_fn,
            desc='ratio_no_batch_dim'),
        ModuleInput(
            constructor_input=FunctionInput((2, 2, 2), output_size=(4, 4, 4), _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((4, 7, 7, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='size_no_batch_dim'),
    ]


def module_inputs_torch_nn_Sigmoid(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Sigmoid: scalar, unbatched, and 4D / 5D inputs
    # labelled as channels-last memory format samples.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(())),
            desc='scalar'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 4, 5))),
            desc='channels_last_mem_format'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
            desc='channels_last_3d_mem_format'
        )
    ]


def module_inputs_torch_nn_LogSigmoid(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LogSigmoid: scalar and 3D inputs with a
    # log-of-sigmoid reference, and an unbatched input.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(())),
            reference_fn=lambda m, p, i: i.sigmoid().log(),
            desc='scalar'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 4))),
            reference_fn=lambda m, p, i: i.sigmoid().log(),
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        ),
    ]


def module_inputs_torch_nn_MarginRankingLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MarginRankingLoss: cases for default, sum, mean,
    # none reduction and a margin of 0.5 on 50-element inputs with sign targets; the
    # reference is marginrankingloss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('margin', {'margin': 0.5})
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i1, i2, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters
            # and returns the result of marginrankingloss_reference for the given input(s) and
            # the bound case options. The case options are bound as defaults so each sample
            # keeps its own.
            return marginrankingloss_reference(i1, i2, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((50,)), make_input((50,)),
                                                    make_target((50,)).sign()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_MultiLabelMarginLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MultiLabelMarginLoss: cases for default, sum, mean
    # and none reduction on a 1D 10-element and a 5x10 input with random integer targets; the
    # reference is multilabelmarginloss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of multilabelmarginloss_reference for the given input(s) and
            # the bound case options. The case options are bound as defaults so each sample
            # keeps its own.
            return multilabelmarginloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((10,)),
                                                    make_target((10), low=0, high=10)),
                        desc=f'1d_{desc}',
                        reference_fn=reference_fn)
        )

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_target((5, 10), low=0, high=10)),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_MultiMarginLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MultiMarginLoss: cases for default, sum, mean, none
    # reduction, p of 2, margin 0.5 and class weights on a 5x10 input with class-index targets;
    # the reference is multimarginloss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('p', {'p': 2}),
        ('margin', {'margin': 0.5}),
        ('weights', {'weight': make_weight(10)})
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of multimarginloss_reference for the given input(s) and the bound
            # case options. The case options are bound as defaults so each sample keeps its own.
            return multimarginloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_target((5), low=0, high=10)),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_error_inputs_torch_nn_MultiMarginLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.MultiMarginLoss: one forward-stage sample where
    # the target length differs from the batch size (regression test for an unclear message).
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    """
    Error inputs for MultiMarginLoss that test the improved error message
    for inconsistent target size.
    Regression test for issue #106251.
    """
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        # Test: target size doesn't match batch size (5 vs 3)
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(),
                forward_input=FunctionInput(
                    make_input((5, 10)),  # input: batch_size=5, num_classes=10
                    torch.tensor([0, 1, 2], device=device, dtype=torch.long),  # target: wrong size (3)
                ),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex=r"target tensor should be 1-D with size equal to.*Expected target size \[5\].*but got \[3\]"
        ),
    ]


def module_inputs_torch_nn_MultiLabelSoftMarginLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MultiLabelSoftMarginLoss: cases for default, sum,
    # mean, none reduction and weights on 5x10 inputs with 0/1 targets; a nested reference
    # computes the expected loss.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('weight', {'weight': make_weight(10)}),
    ]

    def multilabelsoftmargin_loss_reference_fn(m, p, i, t, reduction='mean', weight=None):
        # [FUNCTION DOC] HELPER: Expected multilabel soft margin loss: per-element log-sigmoid
        # terms, optionally scaled by weight, summed over the class dim and divided by the class
        # count, then reduced by none, mean or sum.
        result = t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()
        if weight is not None:
            result *= weight
        result = (-result).sum(i.dim() - 1) / i.size(-1)

        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.mean()
        else:
            return result.sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_target((5, 10), low=0, high=2)),
                        desc=desc,
                        reference_fn=partial(multilabelsoftmargin_loss_reference_fn, **constructor_kwargs))
        )

    return module_inputs


def module_inputs_torch_nn_SoftMarginLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.SoftMarginLoss: cases for default, sum,
    # mean and none reduction on 5x5 inputs with plus/minus one targets; the
    # reference is softmarginloss_reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            # [FUNCTION DOC] HELPER: Sample reference: ignores the module and its parameters and
            # returns the result of softmarginloss_reference for the given input(s) and the bound
            # case options. The case options are bound as defaults so each sample keeps its own.
            return softmarginloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 5)),
                                                    make_target((5, 5)).sign()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_TransformerEncoder(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.TransformerEncoder: reuses every
    # TransformerEncoderLayer sample: each layer sample is turned into a real encoder layer
    # (device and dtype injected) wrapped by a two-layer encoder, and a source-mask keyword is
    # renamed to the mask keyword that the encoder expects.
    # KNOWN ISSUE: it mutates the keyword dictionaries of the reused layer samples in place,
    # ignores extra keyword options, passes no module_info to the layer generator, and drops the
    # layer samples' reference functions.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    # Reuse the TransformerEncoderLayer samples since the forward args are nearly the same.
    samples = []
    for layer_module_input in module_inputs_torch_nn_TransformerEncoderLayer(
            None, device, dtype, requires_grad, training):
        # Construct a TransformerEncoderLayer object to pass to TransformerEncoder.
        l_args, l_kwargs = (layer_module_input.constructor_input.args,
                            layer_module_input.constructor_input.kwargs)
        l_kwargs['device'] = device
        l_kwargs['dtype'] = dtype
        encoder_layer = torch.nn.TransformerEncoderLayer(*l_args, **l_kwargs)
        num_layers = 2
        # Note: TransformerEncoderLayer takes a "src_mask" while
        # TransformerEncoder takes a "mask"; rename kwarg appropriately.
        forward_input = layer_module_input.forward_input
        if 'src_mask' in forward_input.kwargs:
            forward_input.kwargs['mask'] = forward_input.kwargs['src_mask']
            del forward_input.kwargs['src_mask']
        samples.append(ModuleInput(
            constructor_input=FunctionInput(encoder_layer, num_layers),
            forward_input=forward_input,
            desc=layer_module_input.desc
        ))
    return samples

def module_inputs_torch_nn_TransformerEncoderLayer(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.TransformerEncoderLayer: three basic samples
    # (default relu, gelu, no bias), then an unbatched grid over source mask, key padding mask,
    # norm_first, batch_first and bias checked with the no-batch-dim reference; when training is
    # true it adds fast-path samples whose reference reruns the module in eval and no-grad mode
    # and compares with the train-mode result (no reference when bias is disabled because the
    # fast path does not run).
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    samples = [
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 16, 0.0),
            forward_input=FunctionInput(
                make_input((2, 3, 4))
            ),
            desc='relu_activation'
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, F.gelu),
            forward_input=FunctionInput(
                make_input((2, 3, 4))
            ),
            desc='gelu_activation'
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, bias=False),
            forward_input=FunctionInput(
                make_input((2, 3, 4))
            ),
            desc='no_bias'
        ), ]

    # Samples below are for validating the no-batch-dim support.
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    for src_mask, src_key_padding_mask, norm_first, batch_first, bias in \
            itertools.product(attn_masks, key_padding_masks, (True, False), (True, False), (True, False)):
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                dropout=0.0, batch_first=batch_first,
                                                norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    make_input((3, 4)), src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=batch_first, kwargs_to_batchify={'src_key_padding_mask': 0}),
                desc=f'no_batch_dim_batch_first_{batch_first}'
            ))

    # Samples below where we pass reference_fn are for validating the fast path,
    # since the fast path requires no_grad mode, we run the fast path in .eval()
    # and no_grad() in the reference_fn and verify that against the results in train mode.
    def fast_path_reference_fn(module, parameters, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Reference that runs the same module in eval mode under no-grad
        # (enabling the inference fast path) and returns that output, then restores train mode.
        # Raises AssertionError if the module is not in training mode on entry.
        if not module.training:
            raise AssertionError("Expected module.training to be True")
        module.train(False)
        with torch.no_grad():
            output = module(*args, **kwargs)
        module.train(True)
        return output

    if training:
        for norm_first, bias in itertools.product((True, False), (True, False)):
            samples.append(
                ModuleInput(
                    constructor_input=FunctionInput(
                        4, 2, 8, dropout=0.0, batch_first=True, norm_first=norm_first, bias=bias
                    ),
                    forward_input=FunctionInput(
                        make_input((2, 3, 4)),
                    ),
                    # fastpath doesn't run when bias=False
                    reference_fn=fast_path_reference_fn if bias else None,
                    desc=f'fastpath_{bias}_norm_first_{norm_first}'
                )
            )

    return samples


def module_inputs_torch_nn_TransformerDecoderLayer(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.TransformerDecoderLayer: three basic samples (relu,
    # gelu, no bias), then a grid over target mask, key padding mask, norm_first, bias and
    # batch_first, producing for each combination an unbatched sample (no-batch-dim reference) and
    # a batched sample whose masks are expanded to the batch.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    samples = [
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 16, 0.0),
            forward_input=FunctionInput(
                make_input((2, 3, 4)), make_input((2, 3, 4))
            ),
            desc='relu_activation'
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, F.gelu),
            forward_input=FunctionInput(
                make_input((2, 3, 4)), make_input((2, 3, 4))
            ),
            desc='gelu_activation'
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, bias=False),
            forward_input=FunctionInput(
                make_input((2, 3, 4)), make_input((2, 3, 4))
            ),
            desc='no_bias'
        ), ]

    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    for tgt_mask, tgt_key_padding_mask, norm_first, bias, batch_first in \
            itertools.product(attn_masks, key_padding_masks, (True, False), (True, False), (True, False)):
        # Using same mask for tgt and memory
        memory_mask = tgt_mask
        memory_key_padding_mask = tgt_key_padding_mask
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                dropout=0.0, batch_first=batch_first,
                                                norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    make_input((3, 4)), make_input((3, 4)), tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=batch_first,
                                     kwargs_to_batchify={'tgt_key_padding_mask': 0, 'memory_key_padding_mask': 0}),
                desc=f'no_batch_dim_batch_first_{batch_first}'
            ))
        src, tgt = make_input((2, 3, 4)), make_input((2, 3, 4))
        if not batch_first:
            src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)
        if tgt_key_padding_mask is not None:
            memory_key_padding_mask, tgt_key_padding_mask = (tgt_key_padding_mask.expand(2, 3),) * 2
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                dropout=0.0, batch_first=batch_first,
                                                norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    src, tgt, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
                ),
                desc=f'norm_first_{norm_first}_batch_first_{batch_first}_bias_{bias}'
            ))

    return samples


def module_inputs_torch_nn_Transformer(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Transformer: a grid over mask, key padding mask,
    # norm_first, bias and batch_first with one encoder and one decoder layer; each combination
    # yields an unbatched sample (no-batch-dim reference) and a batched sample whose key padding
    # masks are expanded to the batch.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = []
    # Samples below are for validating the no-batch-dim support.
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    for mask, key_padding_mask, norm_first, bias, batch_first in \
            itertools.product(attn_masks, key_padding_masks, (True, False), (True, False), (True, False)):
        # Using same mask for tgt and memory
        src_mask , tgt_mask = (mask,) * 2
        src_key_padding_mask, tgt_key_padding_mask = (key_padding_mask,) * 2
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                num_encoder_layers=1, num_decoder_layers=1,
                                                dropout=0.0, batch_first=batch_first, norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    make_input((3, 4)), make_input((3, 4)), tgt_mask=tgt_mask, src_mask=src_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, src_key_padding_mask=src_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=batch_first,
                                     kwargs_to_batchify={'tgt_key_padding_mask': 0, 'src_key_padding_mask': 0}),
                desc=f'no_batch_dim_batch_first_{batch_first}'
            ))

        src, tgt = make_input((2, 3, 4)), make_input((2, 3, 4))
        if not batch_first:
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
        if key_padding_mask is not None:
            src_key_padding_mask, tgt_key_padding_mask = (key_padding_mask.expand(2, 3),) * 2

        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                num_encoder_layers=1, num_decoder_layers=1,
                                                dropout=0.0, batch_first=batch_first, norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    src, tgt, tgt_mask=tgt_mask, src_mask=src_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, src_key_padding_mask=src_key_padding_mask
                ),
            ))
    return samples


def module_inputs_torch_nn_Embedding(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.Embedding: a 2x3 index input and a discontiguous
    # (expanded) 7x512 index input; indices are random below 4.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_empty = partial(torch.empty, device=device, dtype=torch.long, requires_grad=False)
    return [
        ModuleInput(
            constructor_input=FunctionInput(num_embeddings=4, embedding_dim=3),
            forward_input=FunctionInput(make_empty(2, 3).random_(4))
        ),
        ModuleInput(
            constructor_input=FunctionInput(num_embeddings=4, embedding_dim=3),
            forward_input=FunctionInput(make_empty(1, 512).random_(4).expand(7, 512)),
            desc='discontiguous'
        ),
    ]


def module_error_inputs_torch_nn_Embedding(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.Embedding: an out-of-range index (CPU
    # only, since CUDA asserts in the kernel), float indices, and a negative
    # num_embeddings at construction.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    """
    Error inputs for Embedding that test error messages for invalid inputs.
    """
    samples = []

    # Out of range indices: index exceeds num_embeddings
    # Only test on CPU - CUDA triggers kernel assertion instead of Python exception
    if torch.device(device).type == 'cpu':
        samples.append(
            ErrorModuleInput(
                ModuleInput(
                    constructor_input=FunctionInput(num_embeddings=10, embedding_dim=3),
                    forward_input=FunctionInput(torch.tensor([0, 5, 15], device=device, dtype=torch.long)),
                ),
                error_on=ModuleErrorEnum.FORWARD_ERROR,
                error_type=IndexError,
                error_regex=r"index out of range in self"
            )
        )

    # Float indices: wrong dtype for indices (works on all devices)
    samples.append(
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(num_embeddings=10, embedding_dim=3),
                forward_input=FunctionInput(torch.tensor([1.5, 2.5], device=device, dtype=torch.float32)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex=r"Expected tensor for argument.*indices.*to have.*scalar type.*Long.*Int"
        )
    )

    # Negative num_embeddings (construction error, device-independent)
    samples.append(
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(num_embeddings=-1, embedding_dim=3),
                forward_input=FunctionInput(),
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=RuntimeError,
            error_regex=r"Trying to create tensor with negative dimension"
        )
    )

    return samples



def module_inputs_torch_nn_MultiheadAttention(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.MultiheadAttention: all combinations of bias,
    # add_bias_kv, add_zero_attn, key padding mask and attention mask on unbatched 3x3 inputs, for
    # both batch_first settings, using the MHA-specific no-batch-dim reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = []
    bool_vals = (True, False)
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3, 3)))
    products = itertools.product(bool_vals, bool_vals, bool_vals, key_padding_masks, attn_masks)
    for bias, add_bias_kv, add_zero_attn, key_padding_mask, attn_mask in products:
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(embed_dim=3, num_heads=3, batch_first=True,
                                                bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn),
                forward_input=FunctionInput(make_input((3, 3)), make_input((3, 3)), make_input((3, 3)),
                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask),
                reference_fn=no_batch_dim_reference_mha,
            )
        )
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(embed_dim=3, num_heads=3, batch_first=False,
                                                bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn),
                forward_input=FunctionInput(make_input((3, 3)), make_input((3, 3)), make_input((3, 3)),
                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask),
                reference_fn=partial(no_batch_dim_reference_mha, batch_first=False),
            )
        )

    return samples


def module_inputs_torch_nn_RNN_GRU_Cell(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for RNNCell and GRUCell: unbatched input and hidden state
    # with and without bias, and for RNN (keyword is_rnn true) also the relu nonlinearity; all use
    # the no-batch-dim reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = [
        ModuleInput(
            constructor_input=FunctionInput(5, 10),
            forward_input=FunctionInput(make_input(5), make_input(10)),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput(5, 10, bias=True),
            forward_input=FunctionInput(make_input(5), make_input(10)),
            reference_fn=no_batch_dim_reference_fn,
        )
    ]

    is_rnn = kwargs.get('is_rnn', False)
    if is_rnn:
        # RNN also supports `nonlinearity` argument.
        # `tanh` is the default, so we check with `relu`
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(5, 10, bias=True, nonlinearity='relu'),
                forward_input=FunctionInput(make_input(5), make_input(10)),
                reference_fn=no_batch_dim_reference_fn,
            )
        )

    return samples


def module_inputs_torch_nn_LSTMCell(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for LSTMCell: unbatched input with a hidden/cell pair,
    # with and without bias, using the LSTM-cell no-batch-dim reference. Returns a tuple rather
    # than a list.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = (
        ModuleInput(
            constructor_input=FunctionInput(5, 10),
            forward_input=FunctionInput(make_input(5), (make_input(10), make_input(10))),
            reference_fn=no_batch_dim_reference_lstmcell,
        ),
        ModuleInput(
            constructor_input=FunctionInput(5, 10, bias=True),
            forward_input=FunctionInput(make_input(5), (make_input(10), make_input(10))),
            reference_fn=no_batch_dim_reference_lstmcell,
        ),
    )

    return samples

def make_packed_sequence(inp, batch_sizes):
    # [FUNCTION DOC] HELPER: Packs a padded tensor into a PackedSequence using the given lengths.
    # Temporarily clears requires_grad on the input, packs, then re-applies the original flag to
    # the packed data (the input is left with requires_grad false). Lengths must be sorted in
    # decreasing order. Returns the PackedSequence.
    required_grad = inp.requires_grad
    inp.requires_grad_(False)  # user won't have access to inp so won't be able to get its grads
    seq = pack_padded_sequence(inp, batch_sizes)
    seq.data.requires_grad_(required_grad)
    return seq


def module_inputs_torch_nn_RNN_GRU(module_info, device, dtype, requires_grad, training, with_packed_sequence=False, **kwargs):
    # [FUNCTION DOC] Sample generator for RNN and GRU: requires the keyword is_rnn and an optional
    # with_packed_sequence parameter. Sweeps nonlinearity (RNN only), bias, batch_first and
    # bidirectional, giving an unbatched input without and with initial hidden state, and
    # optionally two packed-sequence samples; all use the RNN/GRU no-batch-dim reference.
    # Raises KeyError if is_rnn is not supplied.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    is_rnn = kwargs['is_rnn']
    nonlinearity = ('relu', 'tanh')
    bias = (False, True)
    batch_first = (False, True)
    bidirectional = (False, True)

    samples = []
    if is_rnn:
        prod_gen = product(nonlinearity, bias, batch_first, bidirectional)
    else:
        prod_gen = product(bias, batch_first, bidirectional)

    for args in prod_gen:
        if is_rnn:
            nl, b, b_f, bidir = args
        else:
            b, b_f, bidir = args

        cons_args = {'input_size': 2, 'hidden_size': 2, 'num_layers': 2,
                     'batch_first': b_f, 'bias': b, 'bidirectional': bidir}
        cons_args_hidden = {'input_size': 2, 'hidden_size': 3, 'num_layers': 2,
                            'batch_first': b_f, 'bias': b, 'bidirectional': bidir}

        if is_rnn:
            cons_args['nonlinearity'] = nl
            cons_args_hidden['nonlinearity'] = nl
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args),
                forward_input=FunctionInput(make_input((3, 2))),
                reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),
            )
        )
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args_hidden),
                forward_input=FunctionInput(make_input((3, 2)), make_input((4 if bidir else 2, 3))),
                reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),
            )
        )
        if with_packed_sequence:
            samples.append(
                ModuleInput(
                    constructor_input=FunctionInput(**cons_args),
                    forward_input=FunctionInput(make_packed_sequence(make_input((5, 2, 2)), torch.tensor([5, 3]))),
                    reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),
                )
            )
            samples.append(
                ModuleInput(
                    constructor_input=FunctionInput(**cons_args),
                    forward_input=FunctionInput(make_packed_sequence(make_input((5, 5, 2)), torch.tensor([5, 3, 3, 2, 2]))),
                    reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),
                )
            )

    return samples


def module_inputs_torch_nn_LSTM(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.LSTM: sweeps bias, batch_first, bidirectional and
    # projection size with two layers, giving an unbatched input without and with an initial
    # (hidden, cell) state, using the LSTM no-batch-dim reference.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    bias = (False, True)
    batch_first = (False, True)
    bidirectional = (False, True)
    proj_sizes = (0, 2)

    samples = []
    prod_gen = product(bias, batch_first, bidirectional, proj_sizes)

    for args in prod_gen:
        b, b_f, bidir, proj_size = args
        hidden_size = 3
        cons_args = {'input_size': 2, 'hidden_size': hidden_size, 'num_layers': 2, 'proj_size': proj_size,
                     'batch_first': b_f, 'bias': b, 'bidirectional': bidir}
        cons_args_hidden = {'input_size': 2, 'hidden_size': hidden_size, 'num_layers': 2, 'proj_size': proj_size,
                            'batch_first': b_f, 'bias': b, 'bidirectional': bidir}

        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args),
                forward_input=FunctionInput(make_input((2, 2))),
                reference_fn=partial(no_batch_dim_reference_lstm, batch_first=b_f),
            )
        )

        h_out = proj_size if proj_size > 0 else hidden_size
        hx = (make_input((4 if bidir else 2, h_out)), make_input((4 if bidir else 2, hidden_size)))
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args_hidden),
                forward_input=FunctionInput(make_input((3, 2)), hx),
                reference_fn=partial(no_batch_dim_reference_lstm, batch_first=b_f),
            )
        )


    return samples



def module_inputs_torch_nn_ReflectionPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ReflectionPad1d: scalar padding on an unbatched input
    # (no-batch-dim reference) and tuple padding on a batched input for 1D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((2, 3))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2)),
            forward_input=FunctionInput(make_input((2, 3, 4))),
        ),
    ]

def module_inputs_torch_nn_ReflectionPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ReflectionPad2d: scalar padding on an unbatched input
    # (no-batch-dim reference) and tuple padding on a batched input for 2D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4, 5))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4)),
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),
        ),
    ]

def module_inputs_torch_nn_ReflectionPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ReflectionPad3d: scalar padding on an unbatched input
    # (no-batch-dim reference) and tuple padding on a batched input for 3D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((2, 3, 4, 5))),
            reference_fn=no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 1, 2, 1, 2)),
            forward_input=FunctionInput(make_input((3, 3, 3, 3, 3))),
        ),
    ]

def module_inputs_torch_nn_ReplicationPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ReplicationPad1d: scalar padding on an unbatched
    # input (no-batch-dim reference) and tuple padding on a batched input for 1D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4))),
            reference_fn=no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2)),
            forward_input=FunctionInput(make_input((3, 4, 5))),
        ),
    ]

def module_inputs_torch_nn_ReplicationPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ReplicationPad2d: scalar padding on an unbatched
    # input (no-batch-dim reference) and tuple padding on a batched input for 2D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4, 5))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4)),
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),
        ),
    ]

def module_inputs_torch_nn_ReplicationPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ReplicationPad3d: scalar padding on an unbatched
    # input (no-batch-dim reference) and tuple padding on a batched input for 3D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4, 5, 6)),
            forward_input=FunctionInput(make_input((3, 4, 5, 6, 7))),
        ),
    ]

def module_inputs_torch_nn_ZeroPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ZeroPad1d: scalar padding on an unbatched input
    # (no-batch-dim reference) and tuple padding on a batched input for 1D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2)),
            forward_input=FunctionInput(make_input((3, 4, 5))),
        ),
    ]

def module_inputs_torch_nn_ZeroPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ZeroPad2d: scalar padding on an unbatched input
    # (no-batch-dim reference) and tuple padding on a batched input for 2D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((1, 2, 3))),
            reference_fn=no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4)),
            forward_input=FunctionInput(make_input((1, 2, 3, 4))),
        ),
    ]

def module_inputs_torch_nn_ZeroPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ZeroPad3d: scalar padding on an unbatched input
    # (no-batch-dim reference) and tuple padding on a batched input for 3D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4, 5, 6)),
            forward_input=FunctionInput(make_input((1, 2, 3, 4, 5))),
        ),
    ]

def module_inputs_torch_nn_ConstantPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ConstantPad1d: scalar padding with a fill value on an
    # unbatched input (no-batch-dim reference) and tuple padding with a fill value on another
    # input for 1D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1, 2),
            forward_input=FunctionInput(make_input((3, 4))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2), 3),
            forward_input=FunctionInput(make_input((3, 4, 5))),
        ),
    ]

def module_inputs_torch_nn_ConstantPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ConstantPad2d: scalar padding with a fill value on an
    # unbatched input (no-batch-dim reference) and tuple padding with a fill value on another
    # input for 2D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1, 3),
            forward_input=FunctionInput(make_input((3, 4, 5))),
            reference_fn=no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4), 5),
            forward_input=FunctionInput(make_input((1, 2, 3, 4))),
        ),
    ]

def module_inputs_torch_nn_ConstantPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.ConstantPad3d: scalar padding with a fill value on an
    # unbatched input (no-batch-dim reference) and tuple padding with a fill value on another
    # input for 3D padding.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1, 3),
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4, 5, 6), 7),
            forward_input=FunctionInput(make_input((1, 2, 1, 2, 1))),
        ),
    ]

def module_inputs_torch_nn_CircularPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.CircularPad1d: scalar padding on an unbatched input,
    # and three tuple paddings (including padding equal to the length) on a 1x2x3 input checked
    # with a nested reference built by concatenating wrapped slices.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def padding1d_circular_ref(inp, pad):
        # [FUNCTION DOC] HELPER: Expected circular padding on the last dim: concatenates the last
        # left-pad entries, the input, and the first right-pad entries. Takes the input and a
        # (left, right) pair.
        # KNOWN ISSUE: a left pad of zero would take the whole tensor because of negative-zero
        # slicing; samples avoid it.
        r""" input:
                [[[0., 1., 2.],
                  [3., 4., 5.]]]
                pad: (1, 2)
                output:
                    [[[2., 0., 1., 2., 0., 1.],
                      [5., 3., 4., 5., 3., 4.]]]
            """
        return torch.cat([inp[:, :, -pad[0]:], inp, inp[:, :, :pad[1]]], dim=2)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4))),
            reference_fn=no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2)),
            forward_input=FunctionInput(make_input((1, 2, 3))),
            reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding),
        ),
        ModuleInput(
            constructor_input=FunctionInput((3, 1)),
            forward_input=FunctionInput(make_input((1, 2, 3))),
            reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding),
        ),
        ModuleInput(
            constructor_input=FunctionInput((3, 3)),
            forward_input=FunctionInput(make_input((1, 2, 3))),
            reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding),
        ),
    ]

def module_inputs_torch_nn_CircularPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.CircularPad2d: scalar padding on an unbatched input,
    # and three tuple paddings on small 4D inputs checked with a nested reference that wraps
    # height then width.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def padding2d_circular_ref(inp, pad):
        # [FUNCTION DOC] HELPER: Expected circular padding on the last two dims: wraps the
        # height using the top and bottom pads, then the width using the left and right pads of
        # the four-element padding. Zero pads would misbehave due to negative-zero slicing;
        # samples avoid it.
        r"""input:
                [[[[0., 1., 2],
                   [3., 4., 5.]]]]
                pad: (1, 2, 2, 1)
        output:
            [[[[2., 0., 1., 2., 0., 1.],
               [5., 3., 4., 5., 3., 4.],
               [2., 0., 1., 2., 0., 1.],
               [5., 3., 4., 5., 3., 4.],
               [2., 0., 1., 2., 0., 1.]]]]
        """
        inp = torch.cat([inp[:, :, -pad[2]:], inp, inp[:, :, :pad[3]]], dim=2)
        return torch.cat([inp[:, :, :, -pad[0]:], inp, inp[:, :, :, :pad[1]]], dim=3)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4, 5))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 2, 1)),
            forward_input=FunctionInput(make_input((1, 1, 2, 3))),
            reference_fn=lambda m, p, i: padding2d_circular_ref(i, m.padding),
        ),
        ModuleInput(
            constructor_input=FunctionInput((2, 3, 2, 2)),
            forward_input=FunctionInput(make_input((1, 1, 2, 3))),
            reference_fn=lambda m, p, i: padding2d_circular_ref(i, m.padding),
        ),
        ModuleInput(
            constructor_input=FunctionInput((3, 3, 3, 1)),
            forward_input=FunctionInput(make_input((1, 1, 3, 3))),
            reference_fn=lambda m, p, i: padding2d_circular_ref(i, m.padding),
        ),
    ]

def module_inputs_torch_nn_CircularPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Sample generator for nn.CircularPad3d: scalar padding on an unbatched input,
    # and three tuple paddings on small 5D inputs checked with a nested reference that wraps
    # depth, height then width.
    # Standard sample-function signature (module_info, device, dtype, requires_grad, training,
    # extra keyword options); returns a list of ModuleInput. Only creates random tensors.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)


    def padding3d_circular_ref(inp, pad):
        # [FUNCTION DOC] HELPER: Expected circular padding on the last three dims: wraps depth,
        # height, then width using the six-element padding. Zero pads would misbehave due to
        # negative-zero slicing; samples avoid it.
        r"""input:
                [[[[[ 0.,  1.,  2.],
                    [ 3.,  4.,  5.]],
                   [[ 6.,  7.,  8.],
                    [ 9., 10., 11.]]]]]
            pad: (1, 2, 2, 1, 1, 2)
            output: [[[[[ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.]],

                       [[ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.]],

                       [[ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.]],

                       [[ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.]],

                       [[ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.]]]]]
        """
        inp = torch.cat([inp[:, :, -pad[4]:], inp, inp[:, :, :pad[5]]], dim=2)
        inp = torch.cat([inp[:, :, :, -pad[2]:], inp, inp[:, :, :, :pad[3]]], dim=3)
        return torch.cat([inp[:, :, :, :, -pad[0]:], inp, inp[:, :, :, :, :pad[1]]], dim=4)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 1, 2, 1, 2)),
            forward_input=FunctionInput(make_input((1, 1, 2, 2, 3))),
            reference_fn=lambda m, p, i: padding3d_circular_ref(i, m.padding)
        ),
        ModuleInput(
            constructor_input=FunctionInput((3, 2, 2, 1, 1, 2)),
            forward_input=FunctionInput(make_input((1, 1, 2, 2, 3))),
            reference_fn=lambda m, p, i: padding3d_circular_ref(i, m.padding)
        ),
        ModuleInput(
            constructor_input=FunctionInput((3, 3, 2, 1, 2, 2)),
            forward_input=FunctionInput(make_input((1, 1, 2, 2, 3))),
            reference_fn=lambda m, p, i: padding3d_circular_ref(i, m.padding)
        ),
    ]


# All these operators share similar issues on cuDNN and MIOpen
rnn_gru_lstm_module_info_decorators = (
    # RuntimeError: Batching rule not implemented for aten::_cudnn_rnn_backward.
    # We could not generate a fallback
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_grad",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # NotImplementedError: the derivative for '_cudnn_rnn_backward' is not implemented.
    # Double backwards is not supported for CuDNN RNNs due to limitations in the CuDNN API
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_gradgrad",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # CUDNN GRU doesn't accept non-contiguous hx
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_non_contiguous_tensors",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # MIOPEN GRU doesn't accept non-contiguous hx (this is dispatched to miopen only for float).
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_non_contiguous_tensors",
        active_if=(TEST_CUDNN and TEST_WITH_ROCM), dtypes=(torch.float,), device_type='cuda'
    )
)

# Start of module error inputs functions.

def module_error_inputs_torch_nn_NLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.NLLLoss: an empty input with an empty
    # weight whose dtype differs from the input dtype, expecting a forward RuntimeError
    # about scalar types.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    """
    Error inputs for NLLLoss that test weight dtype must match input dtype.
    Regression test for device parity between CPU and CUDA with empty tensors.
    """
    input_t = torch.tensor([], device=device, dtype=dtype).reshape((0, 0))
    weight_dtype = torch.float32 if dtype == torch.float16 else torch.float16
    weight_t = torch.tensor([], device=device, dtype=weight_dtype)
    target_t = torch.tensor([], device=device, dtype=torch.long)

    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(weight=weight_t),
                forward_input=FunctionInput(input_t, target_t),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex=r"expected scalar type \w+ but found \w+"
        ),
    ]

def module_error_inputs_torch_nn_RNN_GRU_Cell(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for RNNCell and GRUCell: six forward-stage
    # errors: wrong input size, wrong hidden size, batch size mismatch, hidden state with
    # too many dimensions, and the same hidden-size error built with relu and tanh as
    # third constructor argument.
    # KNOWN ISSUE: the generator is shared with GRUCell, where a third positional argument is the
    # bias flag, so those two samples do not test a nonlinearity there.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 11), make_input(3, 20)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="input has inconsistent input_size: got 11 expected 10"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), make_input(3, 21)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="hidden0 has inconsistent hidden_size: got 21, expected 20"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), make_input(5, 20)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="Input batch size 3 doesn't match hidden0 batch size 5"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), make_input(3, 1, 1, 20)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="Expected hidden to be 1D or 2D, got 4D instead"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20, 'relu'),
                forward_input=FunctionInput(make_input(3, 10), make_input(3, 21)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="hidden0 has inconsistent hidden_size: got 21, expected 20"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20, 'tanh'),
                forward_input=FunctionInput(make_input(3, 10), make_input(3, 21)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="hidden0 has inconsistent hidden_size: got 21, expected 20"
        ),
    ]
    return samples

def module_error_inputs_torch_nn_LSTMCell(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for LSTMCell: four forward-stage errors: wrong input
    # size, wrong hidden size, batch size mismatch, and hidden state with too many dimensions.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 11), (make_input(3, 20), make_input(3, 20))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="input has inconsistent input_size: got 11 expected 10"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), (make_input(3, 21), make_input(3, 21))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="hidden0 has inconsistent hidden_size: got 21, expected 20"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), (make_input(5, 20), make_input(5, 20))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="Input batch size 3 doesn't match hidden0 batch size 5"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), (make_input(3, 1, 1, 20), make_input(3, 1, 1, 20))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="Expected hx\\[0\\] to be 1D or 2D, got 4D instead"
        ),
    ]
    return samples


def module_error_inputs_torch_nn_RNN_GRU(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for RNN and GRU: construction errors for non-positive
    # hidden_size, num_layers and input_size, wrong types for bias, batch_first, input_size and
    # hidden_size, and one forward error for input dtype differing from weight dtype (the
    # mismatched dtype is float64, or float16 on MPS, when dtype is float32; otherwise float32).
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    # use float64 for dtype mismatch test if current dtype is float32, otherwise use float32
    # MPS doesn't support float64, so use float16 instead
    # Extract device type from device string (e.g., 'mps:0' -> 'mps')
    device_type = device.split(':')[0] if isinstance(device, str) else device.type
    if dtype == torch.float32:
        mismatched_dtype = torch.float16 if device_type == 'mps' else torch.float64
    else:
        mismatched_dtype = torch.float32
    make_input = partial(make_tensor, device=device, dtype=mismatched_dtype, requires_grad=requires_grad)

    samples = [
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(10, 0, 1)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=ValueError,
            error_regex="hidden_size must be greater than zero"
        ),
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(10, 10, 0)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=ValueError,
            error_regex="num_layers must be greater than zero"
        ),
        # Test dtype mismatch error message
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(3, 5, dtype=dtype, device=device),
                forward_input=FunctionInput(make_input((2, 4, 3))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex=(r"RNN input dtype .* does not match weight dtype .* "
                         r"Convert input: input\.to\(.*\), or convert model: model\.to\(.*\)")
        ),
        # Test bias parameter type validation
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(3, 5, bias=0)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="bias should be of type bool, got: int"
        ),
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(3, 5, bias=1)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="bias should be of type bool, got: int"
        ),
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(3, 5, bias="True")),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="bias should be of type bool, got: str"
        ),
        # Test batch_first parameter type validation
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(3, 5, batch_first=0)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="batch_first should be of type bool, got: int"
        ),
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(3, 5, batch_first=1)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="batch_first should be of type bool, got: int"
        ),
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(3, 5, batch_first="False")),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="batch_first should be of type bool, got: str"
        ),
        # Test input_size parameter type validation
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(3.0, 5)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="input_size should be of type int, got: float"
        ),
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput("10", 5)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="input_size should be of type int, got: str"
        ),
        # Test input_size parameter value validation
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(0, 5)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=ValueError,
            error_regex="input_size must be greater than zero"
        ),
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(-1, 5)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=ValueError,
            error_regex="input_size must be greater than zero"
        ),
        # Test hidden_size parameter type validation
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(3, 5.0)),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex="hidden_size should be of type int, got: float"
        ),
    ]
    return samples

def module_error_inputs_torch_nn_Pad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for the 1D padding modules: one forward ValueError for
    # a 4D input (1D padding accepts 2D or 3D). Reads is_constant to add a fill value argument.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    is_constant = kwargs.get('is_constant', False)

    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(1, 3) if is_constant else FunctionInput(3),
                forward_input=FunctionInput(make_input((2, 3, 4, 5))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex=r"expected 2D or 3D input \(got 4D input\)",

        ),
    ]

def module_error_inputs_torch_nn_Pad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for the 2D padding modules: one forward ValueError for
    # a 2D input (2D padding accepts 3D or 4D). Reads is_constant to add a fill value argument.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    is_constant = kwargs.get('is_constant', False)

    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(1, 3) if is_constant else FunctionInput(3),
                forward_input=FunctionInput(make_input((2, 3))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex=r"expected 3D or 4D input \(got 2D input\)",

        ),
    ]

def module_error_inputs_torch_nn_Pad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for the 3D padding modules: one forward ValueError for
    # a 2D input (3D padding accepts 4D or 5D). Reads is_constant to add a fill value argument.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    is_constant = kwargs.get('is_constant', False)

    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(1, 3) if is_constant else FunctionInput(3),
                forward_input=FunctionInput(make_input((2, 3))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex=r"expected 4D or 5D input \(got 2D input\)",

        ),
    ]

def module_error_inputs_torch_nn_HuberLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.HuberLoss: one construction TypeError for
    # a complex delta.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(delta=1.0 + 0.0j),
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex=r"delta must be a float or int, got: <class 'complex'>",
        ),
    ]


def module_error_inputs_torch_nn_Softmax(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.Softmax: one construction TypeError for
    # a boolean dim.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(dim=True),
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex=r"dim must be an int or None",
        ),
    ]


def module_error_inputs_torch_nn_LogSoftmax(module_info, device, dtype, requires_grad, training, **kwargs):
    # [FUNCTION DOC] Error-sample generator for nn.LogSoftmax: two construction TypeErrors for a
    # float dim and a boolean dim.
    # Standard sample-function signature; returns a list of ErrorModuleInput, each giving the
    # stage (construction or forward), the expected exception type and a message regex.
    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(dim=1.0),
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex=r"dim must be an int or None",
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(dim=True),
            ),
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=TypeError,
            error_regex=r"dim must be an int or None",
        ),
    ]


_macos15_or_newer = torch.backends.mps.is_available() and torch.backends.mps.is_macos_or_newer(15, 0)


# Database of ModuleInfo entries in alphabetical order.
module_db: list[ModuleInfo] = [
    ModuleInfo(torch.nn.AdaptiveAvgPool1d,
               module_inputs_func=module_inputs_torch_nn_AdaptiveAvgPool1d,
               skips=(
                   # Fails on MPS backend if input/output sizes are not divisible
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.AdaptiveAvgPool2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveAvgPool2d,
               skips=(
                   # Fails on MPS backend if input/output sizes are not divisible
                   DecorateInfo(skipMPS),
                   # Fails on backward check if output size is 1x1
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                   ),)
               ),
    ModuleInfo(torch.nn.AdaptiveAvgPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveAvgPool3d,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # not supported on MPS backend
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.AdaptiveMaxPool1d,
               module_inputs_func=module_inputs_torch_nn_AdaptiveMaxPool1d,
               ),
    ModuleInfo(torch.nn.AdaptiveMaxPool2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveMaxPool2d,
               ),
    ModuleInfo(torch.nn.AdaptiveMaxPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveMaxPool3d,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # not supported on MPS backend
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.AvgPool1d,
               module_inputs_func=module_inputs_torch_nn_AvgPool1d,
               ),
    ModuleInfo(torch.nn.AvgPool2d,
               module_inputs_func=module_inputs_torch_nn_AvgPool2d,
               ),
    ModuleInfo(torch.nn.AvgPool3d,
               module_inputs_func=module_inputs_torch_nn_AvgPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # No channels_last support for AvgPool1d as it does not take 4D inputs
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # backward not supported on MPS backend
                   DecorateInfo(skipMPS, 'TestModule', 'test_non_contiguous_tensors'),)
               ),
    ModuleInfo(torch.nn.BatchNorm1d,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_BatchNorm1d,
               module_error_inputs_func=module_error_inputs_torch_nn_BatchNorm1d_2d_3d,
               skips=(
                   # tracking here rather than in the list in test_aotdispatch.py as eval mode passes
                   # RuntimeError: tried to get Double out of SymInt
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_symbolic_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),
                   # torch._subclasses.fake_tensor.DataDependentOutputException: aten._local_scalar_dense.default
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ))
               ),
    ModuleInfo(torch.nn.BatchNorm2d,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_BatchNorm2d,
               module_error_inputs_func=module_error_inputs_torch_nn_BatchNorm1d_2d_3d,
               skips=(
                   # tracking here rather than in the list in test_aotdispatch.py as eval mode passes
                   # RuntimeError: tried to get Double out of SymInt
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_symbolic_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),
                   # torch._subclasses.fake_tensor.DataDependentOutputException: aten._local_scalar_dense.default
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),)
               ),
    ModuleInfo(torch.nn.BatchNorm3d,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_BatchNorm3d,
               module_error_inputs_func=module_error_inputs_torch_nn_BatchNorm1d_2d_3d,
               skips=(
                   # not supported on MPS backend
                   DecorateInfo(skipMPS),
                   # tracking here rather than in the list in test_aotdispatch.py as eval mode passes
                   # RuntimeError: tried to get Double out of SymInt
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_symbolic_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),
                   # torch._subclasses.fake_tensor.DataDependentOutputException: aten._local_scalar_dense.default
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),)
               ),
    ModuleInfo(torch.nn.CELU,
               module_inputs_func=module_inputs_torch_nn_CELU,
               # not MPS specific, will be xfailed for all devices in next PR
               skips=(
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_check_inplace',
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.Conv1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=False),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.Conv2d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=False),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='cuda', dtypes=[torch.float64]),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.Conv3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=False),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # Conv3d is not supported on MPS backend
                   DecorateInfo(skipMPS, device_type="mps"),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.ConvTranspose1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=False, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               dtypes=floating_and_complex_types_and(torch.chalf),
               skips=(
                   # Not implemented for chalf on CPU
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
                   DecorateInfo(precisionOverride({torch.chalf: 5e-03}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.ConvTranspose2d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=False, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               dtypes=floating_and_complex_types_and(torch.chalf),
               skips=(
                   # Fails on backward check because ViewAsRealBackward apply contiguous for grad
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_memory_format',
                                dtypes=(torch.complex32, torch.complex64, torch.complex128)),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.float64, torch.complex128]),
                   # Not implemented for chalf on CPU
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   DecorateInfo(skipIfXpu, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='xpu'),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
                   DecorateInfo(precisionOverride({torch.chalf: 5e-03}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.ConvTranspose3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=False, transposed=True),
               dtypes=floating_and_complex_types_and(torch.chalf),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # ConvTranspose3d is not supported on MPS backend
                   DecorateInfo(skipMPS),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
                   # Not implemented for chalf on CPU
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   DecorateInfo(skipIfXpu, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='xpu'),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
                   DecorateInfo(precisionOverride({torch.complex64: 1e-04}), 'TestModule', 'test_cpu_gpu_parity'),
                   DecorateInfo(precisionOverride({torch.chalf: 5e-03}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.CosineEmbeddingLoss,
               module_inputs_func=module_inputs_torch_nn_CosineEmbeddingLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.ELU,
               module_inputs_func=module_inputs_torch_nn_ELU,
               # not MPS specific, will be xfailed for all devices in next PR
               skips=(
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_check_inplace',
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.FractionalMaxPool2d,
               module_inputs_func=module_inputs_torch_nn_FractionalMaxPool2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # not supported on MPS backend
                   DecorateInfo(skipMPS),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.FractionalMaxPool3d,
               module_inputs_func=module_inputs_torch_nn_FractionalMaxPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # not supported on MPS backend
                   DecorateInfo(skipMPS),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.L1Loss,
               module_inputs_func=module_inputs_torch_nn_L1Loss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.SmoothL1Loss,
               module_inputs_func=module_inputs_torch_nn_SmoothL1Loss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # See #119108: input types 'tensor<f32>' and 'tensor<15x10xf16>' are not broadcast compatible
                   # NS: Still fails on MacOS15.1
                   DecorateInfo(skipIfMPS, 'TestModule', 'test_non_contiguous_tensors',
                                dtypes=[torch.float16], device_type='mps'),),
               ),
    ModuleInfo(torch.nn.LazyConv1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConv2d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='cuda', dtypes=[torch.float64]),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConv3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   # LazyConv3d is not supported on MPS backend
                   DecorateInfo(skipMPS),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConvTranspose1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=True, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConvTranspose2d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=True, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.float64]),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConvTranspose3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=True, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   # LazyConvTranspose3d is not supported on MPS backend
                   DecorateInfo(skipMPS),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.Linear,
               module_inputs_func=module_inputs_torch_nn_Linear,
               skips=(
                   # No channels_last support for Linear currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.Bilinear,
               module_inputs_func=module_inputs_torch_nn_Bilinear,
               decorators=[
                   DecorateInfo(
                       toleranceOverride({
                           torch.float32: tol(atol=1e-4, rtol=1e-4),
                           torch.float64: tol(atol=1e-4, rtol=1e-4)}),
                       'TestModule', 'test_forward', device_type='cpu'),
               ],
               skips=(
                   # No channels_last support for Bilinear currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # See #119108: tolerance issue
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.LPPool1d,
               module_inputs_func=module_inputs_torch_nn_LPPool1d,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),)
               ),
    ModuleInfo(torch.nn.LPPool2d,
               module_inputs_func=module_inputs_torch_nn_LPPool2d,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),
                   # Fails on backward check on MPS
                   # See https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training') and not _macos15_or_newer,
                       device_type='mps',
                   ),)
               ),
    ModuleInfo(torch.nn.LPPool3d,
               module_inputs_func=module_inputs_torch_nn_LPPool3d,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMPS, device_type='mps'),)
               ),
    ModuleInfo(torch.nn.MaxPool1d,
               module_inputs_func=module_inputs_torch_nn_MaxPool1d,
               ),
    ModuleInfo(torch.nn.MaxPool2d,
               module_inputs_func=module_inputs_torch_nn_MaxPool2d,
               module_error_inputs_func=module_error_inputs_torch_nn_MaxPool2d,
               ),
    ModuleInfo(torch.nn.MaxPool3d,
               module_inputs_func=module_inputs_torch_nn_MaxPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               ),
    ModuleInfo(torch.nn.KLDivLoss,
               module_inputs_func=module_inputs_torch_nn_KLDivLoss,
               decorators=(
                   # `mean` reduction over softmaxed inputs accumulates ~1 ULP per
                   # element in fp16; the default fp16 tolerance (atol=1e-5,
                   # rtol=1e-3) is tight enough that small RNG perturbations push
                   # a few elements just past the threshold.
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-3, rtol=5e-3)}),
                                "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # https://github.com/pytorch/pytorch/issues/115588
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_cpu_gpu_parity'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),)
               ),
    ModuleInfo(torch.nn.MSELoss,
               module_inputs_func=module_inputs_torch_nn_MSELoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # See #119108: tolerance issue
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.MarginRankingLoss,
               module_inputs_func=module_inputs_torch_nn_MarginRankingLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.MultiLabelMarginLoss,
               module_inputs_func=module_inputs_torch_nn_MultiLabelMarginLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 'aten::multilabel_margin_loss_forward' is not currently implemented for the MPS device.
                   DecorateInfo(skipIfMPS, 'TestModule', device_type='mps'),
                   # derivative for aten::multilabel_margin_loss_backward is not implemented
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),)
               ),
    ModuleInfo(torch.nn.MultiMarginLoss,
               module_inputs_func=module_inputs_torch_nn_MultiMarginLoss,
               module_error_inputs_func=module_error_inputs_torch_nn_MultiMarginLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 'aten::multi_margin_loss' is not currently implemented for the MPS device.
                   DecorateInfo(skipIfMPS, 'TestModule', device_type='mps'),
                   # RuntimeError: derivative for aten::multi_margin_loss_backward is not implemented
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),)
               ),
    ModuleInfo(torch.nn.SoftMarginLoss,
               module_inputs_func=module_inputs_torch_nn_SoftMarginLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # See #119108: tolerance issue
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.MultiLabelSoftMarginLoss,
               module_inputs_func=module_inputs_torch_nn_MultiLabelSoftMarginLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.NLLLoss,
               module_inputs_func=module_inputs_torch_nn_NLLLoss,
               module_error_inputs_func=module_error_inputs_torch_nn_NLLLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # See #119108: tolerance issue
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.GaussianNLLLoss,
               module_inputs_func=module_inputs_torch_nn_GaussianNLLLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)),
    ModuleInfo(torch.nn.PoissonNLLLoss,
               module_inputs_func=module_inputs_torch_nn_PoissonNLLLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)),
    ModuleInfo(torch.nn.HingeEmbeddingLoss,
               module_inputs_func=module_inputs_torch_nn_HingeEmbeddingLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.HuberLoss,
               module_inputs_func=module_inputs_torch_nn_HuberLoss,
               module_error_inputs_func=module_error_inputs_torch_nn_HuberLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # See #119108: seemingly incorrect output dtype
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.BCELoss,
               module_inputs_func=module_inputs_torch_nn_BCELoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # error: input types 'tensor<f32>' and 'tensor<15x10xf16>' are not broadcast compatible
                   DecorateInfo(skipIfMPS, 'TestModule', dtypes=[torch.float16], device_type='mps'),)
               ),
    ModuleInfo(torch.nn.BCEWithLogitsLoss,
               module_inputs_func=module_inputs_torch_nn_BCEWithLogitsLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # see #119108: tolerance issue
                   DecorateInfo(skipIfMPS, 'TestModule', dtypes=[torch.float16], device_type='mps'),)
               ),
    ModuleInfo(torch.nn.CrossEntropyLoss,
               module_inputs_func=module_inputs_torch_nn_CrossEntropyLoss,
               module_error_inputs_func=module_error_inputs_torch_nn_CrossEntropyLoss,
               dtypes=get_all_fp_dtypes(include_half=True, include_bfloat16=False),
               decorators=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_memory_format'),
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=3e-2, rtol=1e-3)}), "TestModule",
                                "test_forward", dtypes=[torch.float16], device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_cpu_gpu_parity", dtypes=[torch.float16],
                                device_type='cuda'),
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_cpu_gpu_parity", dtypes=[torch.float16],
                                device_type='xpu'),),
               ),
    ModuleInfo(torch.nn.LinearCrossEntropyLoss,
               module_inputs_func=module_inputs_torch_nn_LinearCrossEntropyLoss,
               module_error_inputs_func=module_error_inputs_torch_nn_LinearCrossEntropyLoss,
               dtypes=get_all_fp_dtypes(include_half=True, include_bfloat16=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               decorators=(
                   # No channels_last support for loss functions,
                   # requires at least a 3-dimensional loss to
                   # reproduce the failure.
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_memory_format'),
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=2e-3, rtol=1e-2)}), "TestModule",
                                "test_non_contiguous_tensors", dtypes=[torch.float16]),
                   DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=1e-2, rtol=5e-2)}), "TestModule",
                                "test_non_contiguous_tensors", dtypes=[torch.bfloat16]),
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=4e-2, rtol=3e-1)}), "TestModule",
                                "test_cpu_gpu_parity", dtypes=[torch.float16]),
                   # Insufficient accuracy, likely related to an issue with cross_entropy
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_cpu_gpu_parity",
                                dtypes=[torch.bfloat16], device_type='cuda'),
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_cpu_gpu_parity",
                                dtypes=[torch.bfloat16], device_type='xpu'),
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=2e-1, rtol=2e-3)}), "TestModule",
                                "test_save_load", device_type="cuda", dtypes=[torch.float16]),
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=2e-1, rtol=2e-3)}), "TestModule",
                                "test_save_load", device_type="xpu", dtypes=[torch.float16]),
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=2e-3, rtol=2e-3)}), "TestModule",
                                "test_forward", dtypes=[torch.float16]),
                   DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=5e-2, rtol=5e-2)}), "TestModule",
                                "test_forward", dtypes=[torch.bfloat16]),
                   DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=2e-1, rtol=5e-2)}), "TestModule",
                                "test_save_load", device_type="cuda", dtypes=[torch.bfloat16]),
               ),
               skips=(
                   # The chunked reduction='none' backward recomputes grads
                   # via in-place buffer accumulation, which gradcheck's
                   # batched-grad path (vmapped cotangent) cannot handle.
                   # Grad correctness is covered by the fp64 gradcheck and
                   # ULP comparisons in test_nn.py and the OpInfo variants.
                   DecorateInfo(unittest.skip("chunked none backward not batched-grad compatible"),
                                'TestModule', 'test_grad'),
                   DecorateInfo(unittest.skip("jacobian mismatch"), 'TestModule', 'test_gradgrad'),),
               ),
    ModuleInfo(torch.nn.CTCLoss,
               module_inputs_func=module_inputs_torch_nn_CTCLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # The operator aten::_ctc_loss is not currently implemented for the MPS device.
                   DecorateInfo(skipIfMPS, 'TestModule', device_type='mps',),
                   # derivative for aten::_ctc_loss_backward is not implemented
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),
                   # https://github.com/pytorch/pytorch/issues/115585
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_non_contiguous_tensors'),)
               ),
    ModuleInfo(torch.nn.GELU,
               module_inputs_func=module_inputs_torch_nn_GELU,
               skips=(
                   # See #119108: tolerance issue
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.GLU,
               module_inputs_func=module_inputs_torch_nn_GLU,
               ),
    ModuleInfo(torch.nn.GroupNorm,
               module_inputs_func=module_inputs_torch_nn_GroupNorm,
               module_error_inputs_func=module_error_inputs_torch_nn_GroupNorm,
               dtypes=get_all_fp_dtypes(include_bfloat16=True, include_half=True),
               skips=(
                   # Tracking at https://github.com/pytorch/pytorch/issues/98089
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_cpu_gpu_parity'),
                   DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-4, rtol=1e-4)}),
                                'TestModule', 'test_memory_format', device_type='cpu'),
                   # No channels_last support for GroupNorm currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', device_type='cuda'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', device_type='mps'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', device_type='xpu'),
                   DecorateInfo(unittest.skip("Skipped!"), "TestModule", "test_grad",
                                active_if=TEST_WITH_ROCM, device_type='cuda'),)
               ),
    ModuleInfo(torch.nn.Hardshrink,
               module_inputs_func=module_inputs_torch_nn_Hardshrink,
               ),
    ModuleInfo(torch.nn.Hardswish,
               module_inputs_func=module_inputs_torch_nn_Hardswish,
               supports_gradgrad=False),
    ModuleInfo(torch.nn.Hardtanh,
               module_inputs_func=module_inputs_torch_nn_Hardtanh,
               ),
    ModuleInfo(torch.nn.InstanceNorm1d,
               module_inputs_func=partial(module_inputs_torch_nn_InstanceNormNd, N=1),
               train_and_eval_differ=True,
               skips=(
                   # No channels_last support for InstanceNorm1d currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.InstanceNorm2d,
               module_inputs_func=partial(module_inputs_torch_nn_InstanceNormNd, N=2),
               train_and_eval_differ=True,
               skips=(
                   # No channels_last support for InstanceNorm2d currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.InstanceNorm3d,
               module_inputs_func=partial(module_inputs_torch_nn_InstanceNormNd, N=3),
               train_and_eval_differ=True,
               skips=(
                   # not supported on MPS backend
                   DecorateInfo(expectedFailureMPS, 'TestModuleMPS', 'test_memory_format'),
                   DecorateInfo(expectedFailureMPS, 'TestModuleMPS', 'test_non_contiguous_tensors'),
                   DecorateInfo(expectedFailureMPS, 'TestModuleMPS', 'test_forward'),
                   DecorateInfo(expectedFailureMPS, 'TestModuleMPS', 'test_non_contiguous'),
                   DecorateInfo(expectedFailureMPS, 'TestModuleMPS', 'test_save_load'),
                   # No channels_last support for InstanceNorm3d currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.LocalResponseNorm,
               module_inputs_func=module_inputs_torch_nn_LocalResponseNorm,
               ),
    ModuleInfo(torch.nn.LayerNorm,
               module_inputs_func=module_inputs_torch_nn_LayerNorm,
               skips=(
                   # No channels_last support for LayerNorm currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.RMSNorm,
               module_inputs_func=module_inputs_torch_nn_RMSNorm,
               ),
    # TransformerEncoder takes the same inputs as TransformerEncoderLayer
    ModuleInfo(torch.nn.TransformerEncoder,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_TransformerEncoder,
               decorators=[
                   # Not implemented for SDPA backward derivative
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad',
                                device_type='cpu'),
               ],
               skips=(
                   # No channels_last support for TransformerEncoderLayer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # Doesn't support device / dtype kwargs directly because it is just a
                   # container of TransformerEncoderLayers.
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_factory_kwargs'),)
               ),
    ModuleInfo(torch.nn.TransformerEncoderLayer,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_TransformerEncoderLayer,
               decorators=[
                   DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-4, rtol=1e-4)}),
                                'TestModule', 'test_non_contiguous_tensors',
                                device_type='cpu', active_if=IS_WINDOWS),
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-4, rtol=2e-3)}),
                                'TestModule', 'test_forward',
                                device_type='mps'),
                   # Not implemented for SDPA backward derivative
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad',
                                device_type='cpu'),
               ],
               skips=(
                   # No channels_last support for TransformerEncoderLayer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.TransformerDecoderLayer,
               module_inputs_func=module_inputs_torch_nn_TransformerDecoderLayer,
               decorators=[
                   # Not implemented for SDPA backward derivative
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad',
                                device_type='cpu'),
               ],
               skips=(
                   # No channels_last support for TransformerDecoderLayer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.Transformer,
               module_inputs_func=module_inputs_torch_nn_Transformer,
               # Inputs are too large to run with slow gradcheck
               # https://github.com/pytorch/pytorch/issues/117140
               gradcheck_fast_mode=True,
               decorators=[
                   # Not implemented for SDPA backward derivative
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad',
                                device_type='cpu'),
               ],
               skips=(
                   # No channels_last support for Transformer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.MultiheadAttention,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_MultiheadAttention,
               decorators=[
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=2e-1, rtol=1e-3)}),
                                'TestModule', 'test_non_contiguous_tensors',
                                device_type='mps')],
               skips=(
                   # No channels_last support for MultiheadAttention currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.Embedding,
               module_inputs_func=module_inputs_torch_nn_Embedding,
               module_error_inputs_func=module_error_inputs_torch_nn_Embedding,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               decorators=[
                   DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-4, rtol=1e-4)}),
                                'TestModule', 'test_non_contiguous_tensors',
                                device_type='mps')],
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.ReLU,
               module_inputs_func=module_inputs_torch_nn_ReLU,
               skips=None if _macos15_or_newer else (
                   # Fails on backward check on MPS
                   # See https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),)
               ),
    ModuleInfo(torch.nn.LeakyReLU,
               module_inputs_func=module_inputs_torch_nn_LeakyReLU,
               ),
    ModuleInfo(torch.nn.ReLU6,
               module_inputs_func=module_inputs_torch_nn_ReLU6,
               skips=(
                   # test fails on MPS backend and is being investigated.
                   # See https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.PReLU,
               module_inputs_func=module_inputs_torch_nn_PReLU,
               skips=(
                   # test fails on MPS backend and is being investigated.
                   # See https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.RNNCell,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU_Cell, is_rnn=True),
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU_Cell,
               ),
    ModuleInfo(torch.nn.GRUCell,
               module_inputs_func=module_inputs_torch_nn_RNN_GRU_Cell,
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU_Cell,
               ),
    ModuleInfo(torch.nn.LSTMCell,
               module_inputs_func=module_inputs_torch_nn_LSTMCell,
               module_error_inputs_func=module_error_inputs_torch_nn_LSTMCell,
               ),
    ModuleInfo(torch.nn.Sigmoid,
               module_inputs_func=module_inputs_torch_nn_Sigmoid,
               ),
    ModuleInfo(torch.nn.LogSigmoid,
               module_inputs_func=module_inputs_torch_nn_LogSigmoid,
               decorators=(
                   # The test's reference_fn is `i.sigmoid().log()`, which in fp16
                   # loses precision as sigmoid(x) saturates near 1 for moderate x.
                   # log_sigmoid uses the stable `min(0,x) - log1p(exp(-|x|))`
                   # in fp32, so it can differ from the naive reference by up to
                   # ~1 ULP of sigmoid(x) (~4e-3 for x in fp16's mid-range).
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=5e-3, rtol=1e-3)}),
                                "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               ),
    ModuleInfo(torch.nn.SiLU,
               module_inputs_func=module_inputs_torch_nn_SiLU,
               ),
    ModuleInfo(torch.nn.Softmax,
               module_inputs_func=module_inputs_torch_nn_Softmax,
               module_error_inputs_func=module_error_inputs_torch_nn_Softmax,
               ),
    ModuleInfo(torch.nn.Softmax2d,
               module_inputs_func=module_inputs_torch_nn_Softmax2d,
               skips=(
                   # no channels last support for Softmax2d currently
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # See #119108: tolerance issue
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward", device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.LogSoftmax,
               module_inputs_func=module_inputs_torch_nn_LogSoftmax,
               module_error_inputs_func=module_error_inputs_torch_nn_LogSoftmax,
               skips=(
                   # no channels last support for LogSoftmax currently
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # See #119108: inf nan error
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward", device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.Softmin,
               module_inputs_func=module_inputs_torch_nn_Softmin,
               skips=(
                   # no channels last support for Softmin currently
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.Softplus,
               module_inputs_func=module_inputs_torch_nn_Softplus,
               skips=(
                   # test fails on MPS backend and is being investigated.
                   # See https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.Softshrink,
               module_inputs_func=module_inputs_torch_nn_Softshrink,
               skips=(
                   # not supported on MPS backend
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.Softsign,
               module_inputs_func=module_inputs_torch_nn_Softsign,
               ),
    ModuleInfo(torch.nn.Tanh,
               module_inputs_func=module_inputs_torch_nn_Tanh,
               skips=None if _macos15_or_newer else (
                   # Fails on backward check on MPS
                   # See https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),)
               ),
    ModuleInfo(torch.nn.Tanhshrink,
               module_inputs_func=module_inputs_torch_nn_Tanhshrink,
               skips=None if _macos15_or_newer else (
                   # Fails on backward check on MPS
                   # See https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),)
               ),
    ModuleInfo(torch.nn.Threshold,
               module_inputs_func=module_inputs_torch_nn_Threshold,
               skips=(
                   # test fails on MPS backend and is being investigated.
                   # See https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.Mish,
               module_inputs_func=module_inputs_torch_nn_Mish,
               skips=(
                   # not supported on MPS backend
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.RNN,
               train_and_eval_differ=True,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU, is_rnn=True),
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU,
               decorators=rnn_gru_lstm_module_info_decorators
               ),
    ModuleInfo(torch.nn.GRU,
               train_and_eval_differ=True,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU, is_rnn=False),
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU,
               decorators=rnn_gru_lstm_module_info_decorators),
    ModuleInfo(torch.nn.LSTM,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_LSTM,
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU,
               skips=(
                   # LSTM with projections is not currently supported with MPS
                   DecorateInfo(skipMPS),),
               decorators=rnn_gru_lstm_module_info_decorators),
    ModuleInfo(torch.nn.ReflectionPad1d,
               module_inputs_func=module_inputs_torch_nn_ReflectionPad1d,
               ),
    ModuleInfo(torch.nn.ReflectionPad2d,
               module_inputs_func=module_inputs_torch_nn_ReflectionPad2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='cuda'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='mps'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='xpu'),)
               ),
    ModuleInfo(torch.nn.ReflectionPad3d,
               module_inputs_func=module_inputs_torch_nn_ReflectionPad3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='cuda'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='mps'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='xpu'),)
               ),
    ModuleInfo(torch.nn.ReplicationPad1d,
               module_inputs_func=module_inputs_torch_nn_ReplicationPad1d,
               ),
    ModuleInfo(torch.nn.ReplicationPad2d,
               module_inputs_func=module_inputs_torch_nn_ReplicationPad2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='cuda'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='mps'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='xpu'),)
               ),
    ModuleInfo(torch.nn.ReplicationPad3d,
               module_inputs_func=module_inputs_torch_nn_ReplicationPad3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='cuda'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='mps'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='xpu'),)
               ),
    ModuleInfo(torch.nn.SELU,
               module_inputs_func=module_inputs_torch_nn_SELU,
               skips=(
                   # test fails on MPS backend and is being investigated.
                   # See https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.ZeroPad1d,
               module_inputs_func=module_inputs_torch_nn_ZeroPad1d,
               ),
    ModuleInfo(torch.nn.ZeroPad2d,
               module_inputs_func=module_inputs_torch_nn_ZeroPad2d,
               skips=(
                   # Fails with channels last test on MPS backend
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='mps'),)
               ),
    ModuleInfo(torch.nn.ZeroPad3d,
               module_inputs_func=module_inputs_torch_nn_ZeroPad3d,
               skips=(
                   # Fails with channels last test on MPS backend
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='mps'),)
               ),
    ModuleInfo(torch.nn.CircularPad1d,
               module_inputs_func=module_inputs_torch_nn_CircularPad1d,
               module_error_inputs_func=module_error_inputs_torch_nn_Pad1d,
               ),
    ModuleInfo(torch.nn.CircularPad2d,
               module_inputs_func=module_inputs_torch_nn_CircularPad2d,
               module_error_inputs_func=module_error_inputs_torch_nn_Pad2d,
               ),
    ModuleInfo(torch.nn.CircularPad3d,
               module_inputs_func=module_inputs_torch_nn_CircularPad3d,
               module_error_inputs_func=module_error_inputs_torch_nn_Pad3d,
               skips=(
                   # Fails with channels last test on MPS backend
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),)
               ),
    ModuleInfo(torch.nn.ConstantPad1d,
               module_inputs_func=module_inputs_torch_nn_ConstantPad1d,
               ),
    ModuleInfo(torch.nn.ConstantPad2d,
               module_inputs_func=module_inputs_torch_nn_ConstantPad2d,
               skips=(
                   # Fails with channels last test on MPS backend
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='mps'),)
               ),
    ModuleInfo(torch.nn.ConstantPad3d,
               module_inputs_func=module_inputs_torch_nn_ConstantPad3d,
               skips=(
                   # Fails with channels last test on MPS backend
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='mps'),)
               )
]
