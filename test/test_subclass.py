# Owner(s): ["module: nn"]
# [MODULE DOC] Tests that tensor subclasses from the shared subclass database (including the
# base tensor class) behave correctly as parameters and inside modules: parameter invariants,
# deepcopy, serialization, repr, type propagation, optimization, parametrization, lazy modules,
# and storage access restrictions. Defines a parametrize decorator over all database entries,
# and test class instantiation happens at import via the parametrized-tests helper.

import tempfile
from copy import deepcopy
from functools import partial
from unittest import expectedFailure

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.utils.parametrize import (
    register_parametrization,
    remove_parametrizations,
)
from torch.testing._internal.common_subclass import (
    DiagTensorBelow,
    subclass_db,
)
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
)
from torch.testing._internal.logging_tensor import LoggingTensor
from torch.utils._pytree import tree_map

# The current test methodology in this file is to test a variety of real use cases
# with a set of fully-fledged tensor subclasses. In the future, this may change
# to more narrowly specify toy subclasses for each of the specific invariants under
# test, avoiding the need to maintain the set of fully-fledged tensor subclasses.


# Decorator for parametrizing tests across the various tensor classes.
parametrize_tensor_cls = parametrize("tensor_cls", [
    subtest(tensor_cls, name=info.name) for tensor_cls, info in subclass_db.items()])


class TestSubclass(TestCase):
    # [CLASS DOC] Parametrized test case run once per tensor subclass in the subclass
    # database (and, for many tests, once per plain-versus-Parameter wrapping). Holds no
    # state of its own.
    def _create_tensor(self, tensor_cls):
        # [FUNCTION DOC] HELPER: Builds an instance of the given tensor class using the
        # creation function registered for it in the subclass database, with size argument
        # 3. Returns the tensor.
        return subclass_db[tensor_cls].create_fn(3)

    @parametrize_tensor_cls
    @parametrize("tensor_requires_grad", [False, True])
    def test_param_invariants(self, tensor_cls, tensor_requires_grad):
        # [FUNCTION DOC] Checks Parameter construction invariants for a subclass tensor: the
        # result is a Parameter, the requires_grad argument given to the constructor wins over
        # the tensor's own setting, and the source tensor is neither converted into a Parameter
        # nor has its requires_grad changed. Also checks the result is not an instance of an
        # unrelated Parameter subclass.
        x = self._create_tensor(tensor_cls).requires_grad_(tensor_requires_grad)
        param = nn.Parameter(x, requires_grad=(not tensor_requires_grad))

        self.assertIsInstance(param, nn.Parameter)
        # Ensure requires_grad passed to Parameter's constructor takes precedence.
        self.assertEqual(param.requires_grad, not tensor_requires_grad)

        # Ensure original tensor is not mutated by Parameter construction.
        self.assertNotIsInstance(x, nn.Parameter)
        self.assertEqual(x.requires_grad, tensor_requires_grad)

        class UninitializedParam(nn.Parameter):
            # [CLASS DOC] HELPER: Empty Parameter subclass used only as a negative isinstance
            # check target.
            pass

        self.assertNotIsInstance(param, UninitializedParam)

    @skipIfTorchDynamo()
    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_deepcopy(self, tensor_cls, as_param):
        # [FUNCTION DOC] Deep-copies a subclass tensor (optionally wrapped as a Parameter) and
        # checks the copy is equal, a distinct object of the same class, an instance of the
        # tensor class, and still a Parameter when the original was. Skipped under Dynamo.
        x = self._create_tensor(tensor_cls)
        if as_param:
            x = nn.Parameter(x)
        x_copy = deepcopy(x)
        self.assertEqual(x, x_copy)
        self.assertEqual(x.__class__, x_copy.__class__)
        self.assertIsNot(x, x_copy)
        self.assertIsInstance(x_copy, tensor_cls)
        if as_param:
            # Deepcopy should preserve both custom type and "parameter-ness".
            self.assertIsInstance(x_copy, nn.Parameter)

    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_serialization(self, tensor_cls, as_param):
        # [FUNCTION DOC] Saves a subclass tensor (optionally a Parameter) to a temporary file,
        # loads it back with the tensor class allowed as a safe global, and checks equality,
        # distinct identity, the subclass type and Parameter-ness are preserved.
        with tempfile.TemporaryFile() as f:
            x = self._create_tensor(tensor_cls)
            if as_param:
                x = nn.Parameter(x)
            torch.save(x, f)
            f.seek(0)
            with torch.serialization.safe_globals([tensor_cls]):
                x_loaded = torch.load(f)

            self.assertEqual(x, x_loaded)
            self.assertIsNot(x, x_loaded)
            self.assertIsInstance(x_loaded, tensor_cls)
            if as_param:
                # Serialization should preserve both custom type and "parameter-ness".
                self.assertIsInstance(x_loaded, nn.Parameter)

    @skipIfTorchDynamo("Visible only with functorch as functorch monkeypatches tensor str")
    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_repr(self, tensor_cls, as_param):
        # [FUNCTION DOC] Checks the repr string contains the subclass name exactly once (except
        # for the base tensor class) and contains the word Parameter exactly once only when
        # wrapped as a Parameter. Skipped under Dynamo.
        x = self._create_tensor(tensor_cls)
        if as_param:
            x = nn.Parameter(x)
        str_repr = x.__repr__()
        if tensor_cls is not torch.Tensor:
            self.assertEqual(str_repr.count(f"{tensor_cls.__name__}("), 1)
        self.assertEqual(str_repr.count("Parameter"), 1 if as_param else 0)

    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_type_propagation(self, tensor_cls, as_param):
        # [FUNCTION DOC] Adds the subclass tensor (optionally a Parameter) to a plain tensor and
        # checks the result keeps the subclass type only if the database marks the class closed
        # under ops, otherwise is a plain tensor, and in no case is a Parameter.
        x = self._create_tensor(tensor_cls)
        if as_param:
            x = nn.Parameter(x)

        # Call the add operator to produce an output tensor.
        output = x + self._create_tensor(torch.Tensor)

        # Custom type should be propagated across operations if closed under the op, but
        # "parameter-ness" should not be.
        if subclass_db[tensor_cls].closed_under_ops:
            self.assertIsInstance(output, tensor_cls)
        else:
            self.assertIsInstance(output, torch.Tensor)
        self.assertNotIsInstance(output, nn.Parameter)

    @parametrize_tensor_cls
    def test_module_optimization(self, tensor_cls):
        # [FUNCTION DOC] Builds a module holding subclass tensors as a parameter, a parameter
        # list and a parameter dictionary, checks its state dict has eight entries, then runs
        # forward, a backward pass seeded with a scalar tensor, and one SGD step to check
        # optimization works.
        create_fn = partial(self._create_tensor, tensor_cls)

        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module owning one parameter, a list of four parameters and a
            # dictionary of three parameters, all created by the subclass creation function.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameters and initializes them without
                # gradient tracking (normal init for the single one, uniform init for the list
                # and dictionary entries).
                super().__init__()
                self.p1 = nn.Parameter(create_fn())

                self.p_list = nn.ParameterList([create_fn() for _ in range(3)])
                self.p_list.append(create_fn())

                self.p_dict = nn.ParameterDict({
                    'foo': create_fn(),
                    'bar': create_fn(),
                })
                self.p_dict['baz'] = create_fn()

                with torch.no_grad():
                    nn.init.normal_(self.p1)
                    for p in self.p_list:
                        nn.init.uniform_(p)
                    for p in self.p_dict.values():
                        nn.init.uniform_(p)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus every parameter,
                # accumulated in order.
                out = self.p1 + x
                for p in self.p_list:
                    out = p + out

                for v in self.p_dict.values():
                    out = v + out

                return out

        m = MyModule()
        self.assertEqual(len(m.state_dict()), 8)

        optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
        m(create_fn()).sum().backward(torch.tensor(1))
        optimizer.step()

    @parametrize_tensor_cls
    @parametrize("leave_parametrized", [False, True])
    def test_parametrization(self, tensor_cls, leave_parametrized):
        # [FUNCTION DOC] Registers a negation parametrization on the weight of a module holding
        # a subclass tensor, checks the weight and the module output are still the subclass
        # type, then removes the parametrization with or without keeping the parametrized value.
        # Returns immediately (does nothing) for the logging tensor and the diagonal-below
        # classes because their set operation is not supported.
        # TODO: Either implement set_() properly for these tensor subclasses or apply a
        # more general fix to avoid the need for special set_() handling. For now, skip
        # testing these as they're expected to fail.
        if tensor_cls in [LoggingTensor, DiagTensorBelow]:
            return

        create_fn = partial(self._create_tensor, tensor_cls)

        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module with a single weight parameter made from the
            # subclass tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the weight parameter.
                super().__init__()
                self.weight = nn.Parameter(create_fn())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the weight plus the input.
                return self.weight + x

        class MyParametrization(nn.Module):
            # [CLASS DOC] HELPER: Parametrization module that negates its input.
            def forward(self, X):
                # [FUNCTION DOC] HELPER: Returns the negation of the input.
                return -X

        m = MyModule()
        self.assertEqual(len(m.state_dict()), 1)
        register_parametrization(m, 'weight', MyParametrization())
        self.assertIsInstance(m.weight, tensor_cls)
        output = m(self._create_tensor(torch.Tensor))
        self.assertIsInstance(output, tensor_cls)
        remove_parametrizations(m, 'weight', leave_parametrized=leave_parametrized)

    # Lazy modules with custom tensors are not supported yet.
    @expectedFailure
    @parametrize_tensor_cls
    def test_lazy_module(self, tensor_cls):
        # [FUNCTION DOC] Expected-failure test of a lazy module holding an uninitialized
        # parameter being materialized from a subclass tensor. For the base tensor class it
        # fails explicitly with a dummy message; for subclasses it is expected to fail because
        # lazy modules do not support them.
        if tensor_cls is torch.Tensor:
            self.fail('dummy fail for base tensor until the test passes for subclasses')

        class MyLazyModule(LazyModuleMixin, nn.Module):
            # [CLASS DOC] HELPER: Lazy module with one uninitialized parameter that is
            # materialized on first call from the input shape.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the uninitialized parameter.
                super().__init__()
                self.param = nn.UninitializedParameter()

            def initialize_parameters(self, input) -> None:  # type: ignore[override]
                # [FUNCTION DOC] HELPER: If parameters are still uninitialized,
                # materializes the parameter with the input's shape and fills it
                # uniformly, without gradient tracking.
                if self.has_uninitialized_params():
                    with torch.no_grad():
                        self.param.materialize(input.shape)
                        nn.init.uniform_(self.param)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the parameter plus the input.
                return self.param + x

        m = MyLazyModule()
        self.assertTrue(m.has_uninitialized_params())
        m(self._create_tensor(tensor_cls))
        self.assertFalse(m.has_uninitialized_params())
        self.assertIsInstance(m.param, tensor_cls)

    def test_non_rewrapping_torch_dispatch_subclass_as_parameter_throws_for_detach(self):
        # [FUNCTION DOC] Checks that wrapping a dispatch-based subclass whose handler returns
        # unwrapped plain tensors (so detach does not return the same type) in a Parameter
        # raises a RuntimeError saying that detach must return an instance of the same type.

        # Define a subclass that does not rewrap for any function in its __torch_dispatch__ impl.
        class NonRewrappingTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass holding an inner tensor in an
            # attribute; its dispatch handler unwraps arguments and returns the raw result
            # without rewrapping.
            @staticmethod
            def __new__(
                cls, t: torch.Tensor
            ):
                # [FUNCTION DOC] HELPER: Creates a wrapper subclass instance with the shape,
                # dtype, requires_grad and device of the given tensor and returns it.
                r = super()._make_wrapper_subclass(
                    cls, t.shape, dtype=t.dtype, requires_grad=t.requires_grad, device=t.device)
                return r

            def __init__(self, t) -> None:
                # [FUNCTION DOC] HELPER: Stores the wrapped tensor in the tensor attribute.
                self.tensor: torch.Tensor = t

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch handler: replaces every wrapper in the
                # arguments and keyword arguments with its inner tensor, runs the operation and
                # returns the plain result.

                def unwrap(e) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Returns the inner tensor if given a wrapper
                    # instance, otherwise the value unchanged.
                    if isinstance(e, NonRewrappingTensor):
                        t = e.tensor
                        return t
                    else:
                        return e

                r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
                # Return an unwrapped tensor no longer of original subclass type.
                return r

        with self.assertRaisesRegex(RuntimeError, r"requires that detach\(\) returns an instance of the same type"):
            nn.Parameter(NonRewrappingTensor(torch.randn(3)))

    def test_tensor_subclass_storage_data_accesses_throw(self):
        # [FUNCTION DOC] Checks that for a logging tensor subclass the untyped storage can be
        # obtained, but data pointer access, resize, copy, fill and file write on it each raise
        # a RuntimeError mentioning an invalid python storage.
        from torch.testing._internal.logging_tensor import LoggingTensor
        x = torch.ones(2)
        x_log = LoggingTensor(x)
        # Accessing storage on a tensor subclass is valid
        storage = x_log.untyped_storage()
        # This includes accessing metadata on the storage
        # But storage methods that access data will throw
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage.data_ptr()
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage.resize_(0)
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage.copy_(storage)
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage.fill_(0)
        with self.assertRaisesRegex(RuntimeError, "on an invalid python storage"):
            storage._write_file("file")


instantiate_parametrized_tests(TestSubclass)

if __name__ == '__main__':
    run_tests()
