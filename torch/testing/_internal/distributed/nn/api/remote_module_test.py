# mypy: allow-untyped-defs

# [MODULE DOC] Tests for RemoteModule (a module living on another RPC worker) shared by CPU and
# CUDA suites. Defines RPC handler functions run on the destination worker, a creation-mode enum,
# jit interfaces and small module classes, a common fixture base with a helper that builds remote
# modules in both plain and scripted form, and test classes for two workers, three workers and CUDA
# devices. Import-time side effect: creates one global shared Parameter.
import enum

import torch
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch import nn, Tensor
from torch._jit_internal import Future
from torch.distributed.nn import RemoteModule
from torch.distributed.nn.api.remote_module import (
    _REMOTE_MODULE_PICKLED_ATTRIBUTES,
    _RemoteModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TemporaryFileName, TEST_WITH_ROCM
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


_PARAM_VAL = torch.nn.Parameter(torch.ones(1))


# RPC handler for querying the device on the destination worker.
def remote_device(module_rref):
    # [FUNCTION DOC] HELPER: RPC handler run on the destination worker. Given an RRef to a module,
    # returns the device of its first parameter (None if it has no parameters).
    for param in module_rref.local_value().parameters():
        return param.device


# RPC handler for querying __dict__ on the destination worker.
def remote_module_attributes(remote_module):
    # [FUNCTION DOC] HELPER: RPC handler returning the instance attribute mapping of the given
    # remote module as seen on the destination worker.
    return remote_module.__dict__


# RPC handler for running forward on the destination worker.
def remote_forward(remote_module, args):
    # [FUNCTION DOC] HELPER: RPC handler that calls forward of the given remote module with the
    # given tuple of positional arguments and returns the result.
    return remote_module.forward(*args)


# RPC handler for running forward_async on the destination worker.
def remote_forward_async(remote_module, args):
    # [FUNCTION DOC] HELPER: RPC handler that calls forward_async of the remote module and blocks
    # on the returned future, because futures cannot be pickled; returns the result.
    # Since future cannot be pickled and sent over the RPC layer,
    # have to wait and behave just like ``forward_sync``.
    return remote_module.forward_async(*args).wait()


# RPC handler for getting training mode on the destination worker.
def get_remote_training_arg(module_rref):
    # [FUNCTION DOC] HELPER: RPC handler that returns the training flag of the module behind an
    # RRef that is owned by the destination worker.
    return module_rref.local_value().training


class ModuleCreationMode(enum.Enum):
    # [CLASS DOC] HELPER: Enum of the ways a remote module is constructed in tests: from a plain
    # module class, or from a scripted module with a jit interface.
    MODULE_CTOR_WITH_INTERFACE = "module_ctor_with_interface"
    MODULE_CTOR = "module_ctor"


@torch.jit.interface
class MyModuleInterface:
    # [CLASS DOC] HELPER: jit interface describing a module whose forward takes a tensor, an int
    # and a string and returns a tuple of string, int and tensor.
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> tuple[str, int, Tensor]:
        # [FUNCTION DOC] HELPER: Interface method declaration only; has no behavior.
        # pyre-ignore[7]: Pyre and torch.jit.interface don't mix well
        pass


@torch.jit.interface
class RemoteMyModuleInterface:
    # [CLASS DOC] HELPER: jit interface for a remote module, declaring forward and forward_async.
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> tuple[str, int, Tensor]:
        # [FUNCTION DOC] HELPER: Interface method declaration only; has no behavior.
        # pyre-ignore[7]: Pyre and torch.jit.interface don't mix well
        pass

    def forward_async(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Future[tuple[str, int, Tensor]]:
        # [FUNCTION DOC] HELPER: Interface method declaration only for the asynchronous variant
        # that returns a future; has no behavior.
        pass


class MyModule(nn.Module):
    # [CLASS DOC] HELPER: Test module owning the shared global Parameter; its forward returns the
    # inputs reordered as (word, number, tensor).
    def __init__(self, first_arg, first_kwarg=-1):
        # [FUNCTION DOC] HELPER: Initializes the module and registers the shared global parameter.
        # The constructor arguments are accepted but ignored.
        super().__init__()
        self.param1 = _PARAM_VAL

    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> tuple[str, int, Tensor]:
        # [FUNCTION DOC] HELPER: Returns a tuple of the word, the number and the tensor, in
        # that order.
        return word, number, tensor


class BadModule:
    # [CLASS DOC] HELPER: Deliberately not an nn.Module, used to check that remote module creation
    # rejects non-module classes.
    def __init__(self, first_arg, first_kwarg=-1):
        # [FUNCTION DOC] HELPER: Accepts and ignores its arguments.
        pass


def create_scripted_module(first_arg, first_kwarg=-1):
    # [FUNCTION DOC] HELPER: Builds a MyModule from the arguments and returns it compiled
    # with jit script.
    module = MyModule(first_arg, first_kwarg=first_kwarg)
    scripted_module = torch.jit.script(module)
    return scripted_module


# Common utils for both CPU and CUDA test suites
class CommonRemoteModuleTest(RpcAgentTestFixture):
    # [CLASS DOC] HELPER: Base fixture for the remote module tests, built on the RPC agent
    # test fixture. Provides a world size and a generator of remote modules in the requested
    # creation modes.
    @property
    def world_size(self):  # Override setting in RpcAgentTestFixture
        # [FUNCTION DOC] HELPER: Returns 2, the number of workers used by default.
        return 2

    @staticmethod
    def _create_remote_module_iter(remote_device, modes=None):
        # [FUNCTION DOC] HELPER: Generator that yields remote modules targeting the given remote
        # device string. Modes default to all creation modes. For plain mode it yields a
        # RemoteModule of MyModule; for interface mode it yields a scripted remote module built
        # from a scripted module. Constructor args are (1,) and keyword first_kwarg=2. Performs RPC
        # calls on creation.
        if modes is None:
            modes = ModuleCreationMode.__members__.values()

        args = (1,)
        kwargs = dict(first_kwarg=2)

        if ModuleCreationMode.MODULE_CTOR in modes:
            remote_module = RemoteModule(remote_device, MyModule, args, kwargs)
            yield remote_module

        if ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE in modes:
            remote_module = _RemoteModule(
                remote_device,
                create_scripted_module,
                args,
                kwargs,
                _module_interface_cls=MyModuleInterface,
            )
            scripted_remote_module = torch.jit.script(remote_module)
            yield scripted_remote_module


class RemoteModuleTest(CommonRemoteModuleTest):
    # [CLASS DOC] Tests of RemoteModule behavior with two workers on CPU. Every test runs only on
    # rank 0 and targets the next rank.
    @dist_utils.dist_init
    def test_bad_module(self):
        # [FUNCTION DOC] Checks that creating a RemoteModule from a class that is not an nn.Module
        # raises ValueError with the expected message. Runs on rank 0 only.
        # KNOWN ISSUE: the same check is performed twice with identical code.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        remote_device = f"{dst_worker_name}/cpu"
        args = (1,)
        kwargs = dict(first_kwarg=2)

        with self.assertRaisesRegex(
            ValueError,
            r"Expect `module_cls\(\*args, \*\*kwargs\)` returns an instance of <class nn.Module>,",
        ):
            RemoteModule(remote_device, BadModule, args, kwargs).forward()

        with self.assertRaisesRegex(
            ValueError,
            r"Expect `module_cls\(\*args, \*\*kwargs\)` returns an instance of <class nn.Module>,",
        ):
            RemoteModule(remote_device, BadModule, args, kwargs).forward()

    @dist_utils.dist_init
    def test_forward_async(self):
        # [FUNCTION DOC] Checks that forward_async on remote modules of each creation mode returns
        # a future whose result is the reversed argument tuple. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2, "3")
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret_fut = remote_module.forward_async(*args)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args)))

    @dist_utils.dist_init
    def test_forward_async_script(self):
        # [FUNCTION DOC] Checks forward_async called from inside a scripted function on a scripted
        # remote module returns the expected tuple. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        scripted_remote_module = next(
            self._create_remote_module_iter(
                dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
            )
        )

        @torch.jit.script
        def run_forward_async(scripted_remote_module: RemoteMyModuleInterface):
            # [FUNCTION DOC] HELPER: Scripted function that calls forward_async on the module,
            # waits and returns the result.
            ret_fut = scripted_remote_module.forward_async(torch.ones(1), 2, "3")
            ret = ret_fut.wait()
            return ret

        ret = run_forward_async(scripted_remote_module)

        self.assertEqual(ret, ("3", 2, torch.ones(1)))

    @dist_utils.dist_init
    def test_forward_sync(self):
        # [FUNCTION DOC] Checks that forward on remote modules of each creation mode returns the
        # reversed argument tuple. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2, "3")
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret = remote_module.forward(*args)
            self.assertEqual(ret, tuple(reversed(args)))

    @dist_utils.dist_init
    def test_forward_sync_script(self):
        # [FUNCTION DOC] Checks forward called from inside a scripted function on a scripted remote
        # module returns the expected tuple. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        scripted_remote_module = next(
            self._create_remote_module_iter(
                dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
            )
        )

        @torch.jit.script
        def run_forward(scripted_remote_module: MyModuleInterface):
            # [FUNCTION DOC] HELPER: Scripted function that calls forward on the module and returns
            # the result.
            ret = scripted_remote_module.forward(torch.ones(1), 2, "3")
            return ret

        ret = run_forward(scripted_remote_module)

        self.assertEqual(ret, ("3", 2, torch.ones(1)))

    @dist_utils.dist_init
    def test_forward_with_kwargs(self):
        # [FUNCTION DOC] Checks that forward and forward_async accept keyword arguments for plain
        # remote modules and return the expected tuple. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2)
        kwargs = dict(word="3")
        # Only test Python nn.Module, because script module methods don't support taking kwargs.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            ret_fut = remote_module.forward_async(*args, **kwargs)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args + ("3",))))

            ret = remote_module.forward(*args, **kwargs)
            self.assertEqual(ret, tuple(reversed(args + ("3",))))

    @dist_utils.dist_init
    def test_remote_parameters(self):
        # [FUNCTION DOC] Checks remote_parameters returns one RRef whose fetched value equals the
        # shared parameter. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # Only test Python nn.Module, because script module methods don't support ``remote_parameters``.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            param_rrefs = remote_module.remote_parameters()
            self.assertEqual(len(param_rrefs), 1)
            self.assertTrue(torch.equal(param_rrefs[0].to_here(), _PARAM_VAL))

    @dist_utils.dist_init
    def test_get_module_rref(self):
        # [FUNCTION DOC] Checks get_module_rref equals the module_rref attribute and that the
        # fetched module's parameters equal the shared parameter. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # Only test Python nn.Module, because script module methods don't support ``get_module_rref``.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            rref = remote_module.get_module_rref()
            self.assertEqual(rref, remote_module.module_rref)
            for param in rref.to_here().parameters():
                self.assertTrue(torch.equal(param, _PARAM_VAL))

    @dist_utils.dist_init
    def test_train_eval(self):
        # [FUNCTION DOC] Checks that train and eval on the remote module propagate to the remote
        # worker's training flag (true then false), queried via RPC. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            remote_module.train()
            ret1 = rpc.rpc_sync(
                dst_worker_name,
                get_remote_training_arg,
                args=(remote_module.get_module_rref(),),
            )
            self.assertEqual(ret1, True)

            remote_module.eval()
            ret2 = rpc.rpc_sync(
                dst_worker_name,
                get_remote_training_arg,
                args=(remote_module.get_module_rref(),),
            )
            self.assertEqual(ret2, False)

    @dist_utils.dist_init
    def test_unsupported_methods(self):
        # [FUNCTION DOC] Checks that a long list of nn.Module methods (registration, apply, dtype
        # and device moves, hooks, state dict, parameter, buffer, child and module iteration,
        # requires_grad_, zero_grad, share_memory, extra_repr) each raise ValueError saying the
        # method is not supported. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            with self.assertRaisesRegex(
                ValueError, r"Method ``register_buffer`` not supported for RemoteModule"
            ):
                remote_module.register_buffer("buffer", torch.ones(5))
            with self.assertRaisesRegex(
                ValueError,
                r"Method ``register_parameter`` not supported for RemoteModule",
            ):
                remote_module.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(1))
                )
            with self.assertRaisesRegex(
                ValueError, r"Method ``add_module`` not supported for RemoteModule"
            ):
                remote_module.add_module("empty", None)

            with self.assertRaisesRegex(
                ValueError, r"Method ``apply`` not supported for RemoteModule"
            ):
                fn = torch.rand((3, 3), requires_grad=False)
                remote_module.apply(fn)

            with self.assertRaisesRegex(
                ValueError, r"Method ``cuda`` not supported for RemoteModule"
            ):
                remote_module.cuda()
            with self.assertRaisesRegex(
                ValueError, r"Method ``cpu`` not supported for RemoteModule"
            ):
                remote_module.cpu()
            with self.assertRaisesRegex(
                ValueError, r"Method ``type`` not supported for RemoteModule"
            ):
                remote_module.type(torch.FloatTensor)
            with self.assertRaisesRegex(
                ValueError, r"Method ``float`` not supported for RemoteModule"
            ):
                remote_module.float()
            with self.assertRaisesRegex(
                ValueError, r"Method ``double`` not supported for RemoteModule"
            ):
                remote_module.double()
            with self.assertRaisesRegex(
                ValueError, r"Method ``bfloat16`` not supported for RemoteModule"
            ):
                remote_module.bfloat16()
            with self.assertRaisesRegex(
                ValueError, r"Method ``to`` not supported for RemoteModule"
            ):
                remote_module.to("cpu", dtype=torch.int32)

            def hook(module, grad_input, grad_output):
                # [FUNCTION DOC] HELPER: No-op hook passed to the hook registration methods.
                pass

            with self.assertRaisesRegex(
                ValueError,
                r"Method ``register_backward_hook`` not supported for RemoteModule",
            ):
                remote_module.register_backward_hook(hook)
            with self.assertRaisesRegex(
                ValueError,
                r"Method ``register_forward_pre_hook`` not supported for RemoteModule",
            ):
                remote_module.register_forward_pre_hook(hook)
            with self.assertRaisesRegex(
                ValueError,
                r"Method ``register_forward_hook`` not supported for RemoteModule",
            ):
                remote_module.register_forward_hook(hook)

            with self.assertRaisesRegex(
                ValueError, r"Method ``state_dict`` not supported for RemoteModule"
            ):
                remote_module.state_dict()
            with self.assertRaisesRegex(
                ValueError, r"Method ``load_state_dict`` not supported for RemoteModule"
            ):
                remote_module.load_state_dict({})

            with self.assertRaisesRegex(
                ValueError,
                r"Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead.",
            ):
                remote_module.parameters()
            with self.assertRaisesRegex(
                ValueError,
                r"Method ``named_parameters`` not supported for RemoteModule",
            ):
                remote_module.named_parameters()
            with self.assertRaisesRegex(
                ValueError, r"Method ``buffers`` not supported for RemoteModule"
            ):
                remote_module.buffers()
            with self.assertRaisesRegex(
                ValueError, r"Method ``named_buffers`` not supported for RemoteModule"
            ):
                remote_module.named_buffers()
            with self.assertRaisesRegex(
                ValueError, r"Method ``children`` not supported for RemoteModule"
            ):
                remote_module.children()
            with self.assertRaisesRegex(
                ValueError, r"Method ``named_children`` not supported for RemoteModule"
            ):
                remote_module.named_children()
            with self.assertRaisesRegex(
                ValueError, r"Method ``modules`` not supported for RemoteModule"
            ):
                remote_module.modules()
            with self.assertRaisesRegex(
                ValueError, r"Method ``named_modules`` not supported for RemoteModule"
            ):
                remote_module.named_modules()

            with self.assertRaisesRegex(
                ValueError, r"Method ``requires_grad_`` not supported for RemoteModule"
            ):
                remote_module.requires_grad_()
            with self.assertRaisesRegex(
                ValueError, r"Method ``zero_grad`` not supported for RemoteModule"
            ):
                remote_module.zero_grad()
            with self.assertRaisesRegex(
                ValueError, r"Method ``share_memory`` not supported for RemoteModule"
            ):
                remote_module.share_memory()
            with self.assertRaisesRegex(
                ValueError, r"Method ``extra_repr`` not supported for RemoteModule"
            ):
                remote_module.extra_repr()

    @dist_utils.dist_init
    def test_send_remote_module_with_a_new_attribute_not_pickled_over_the_wire(self):
        # [FUNCTION DOC] Checks that an attribute added after construction is not included when the
        # remote module is sent over RPC, by inspecting the received attribute mapping. Runs on
        # rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # If a new attribute is added to this RemoteModule after the initialization,
        # and it will be sent over the wire by RPC,
        # this new field will not be pickled, because it's not specified in _REMOTE_MODULE_PICKLED_ATTRIBUTES.
        # Note that adding a new attribute out of constructor should rarely happen.
        # If a new attribute is added to RemoteModule constructor,
        # there is a sanity check to enforce developers to add this attribute to either
        # _REMOTE_MODULE_PICKLED_ATTRIBUTES or _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            new_attr_name = "new_attr"
            setattr(remote_module, new_attr_name, 1)

            attrs = rpc.rpc_sync(
                dst_worker_name, remote_module_attributes, (remote_module,)
            )
            self.assertNotIn(new_attr_name, attrs)

    @dist_utils.dist_init
    def test_remote_module_py_pickle_not_supported(self):
        # [FUNCTION DOC] Checks that saving a plain remote module with the Python pickler raises
        # RuntimeError saying it can only be pickled with RPC. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            with TemporaryFileName() as fname:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Cannot pickle RemoteModule in python pickler. RemoteModule can only be pickled when using RPC",
                ):
                    torch.save(remote_module, fname)

    @dist_utils.dist_init
    def test_remote_module_py_pickle_not_supported_script(self):
        # [FUNCTION DOC] Checks that saving a scripted remote module with torch.save raises a jit
        # error saying it can only be pickled with RPC. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
        ):
            with (
                TemporaryFileName() as fname,
                self.assertRaisesRegex(
                    torch.jit.Error, "can only be pickled when using RPC"
                ),
            ):
                torch.save(remote_module, fname)


class ThreeWorkersRemoteModuleTest(CommonRemoteModuleTest):
    # [CLASS DOC] Tests of RemoteModule behavior with three workers, where a module created on one
    # worker is passed to another.
    @property
    def world_size(self):  # Override setting in CommonRemoteModuleTest
        # [FUNCTION DOC] HELPER: Returns 3, the number of workers for this test class.
        return 3

    @dist_utils.dist_init
    def test_send_remote_module_over_the_wire(self):
        # [FUNCTION DOC] Creates a remote module on worker1, sends it to worker2 over RPC and
        # checks the received attribute names, the on and device values and the two flags, then
        # runs forward and forward_async on worker2 and checks the results. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)

        # Unpickled attributes include both the inherent attributes of RemoteModule
        # (not inherited from the superclass) and two installed methods.
        expected_unpickled_attrs = list(_REMOTE_MODULE_PICKLED_ATTRIBUTES)
        expected_unpickled_attrs.append("forward_async")
        expected_unpickled_attrs.append("forward")

        # Create a remote module on worker1 and then pass it to worker2 over the RPC layer.
        for remote_module in self._create_remote_module_iter(
            dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # Test querying some simple attributes from worker2.
            attrs = rpc.rpc_sync(
                dst_worker2_name, remote_module_attributes, (remote_module,)
            )
            self.assertListEqual(list(attrs.keys()), expected_unpickled_attrs)
            self.assertEqual(attrs["on"], "worker1")
            self.assertEqual(attrs["device"], "cpu")
            self.assertFalse(attrs["is_device_map_set"])
            self.assertFalse(attrs["is_scriptable"])

            # Test the installed methods on worker1's can be initiated by worker2 over RPC layer.
            # NOTE: In practice a remote module should be directly stored on the worker that runs ``forward``` or ``forward_async``,
            # not have another worker to initiate forward over the RPC layer.
            args = (torch.ones(1), 2, "3")
            ret1 = rpc.rpc_sync(dst_worker2_name, remote_forward, (remote_module, args))
            self.assertEqual(ret1, tuple(reversed(args)))
            ret2 = rpc.rpc_sync(
                dst_worker2_name, remote_forward_async, (remote_module, args)
            )
            self.assertEqual(ret2, tuple(reversed(args)))

    @dist_utils.dist_init
    def test_send_remote_module_over_the_wire_script_not_supported(self):
        # [FUNCTION DOC] Checks that sending a scripted remote module over RPC raises RuntimeError
        # saying it is not supported. Runs on rank 0 only.
        # KNOWN ISSUE: the local list of expected attribute names is built but never used.
        if self.rank != 0:
            return
        dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)

        # Unpickled attributes include both the inherent attributes of RemoteModule
        # (not inherited from the superclass) and two installed methods.
        expected_unpickled_attrs = list(_REMOTE_MODULE_PICKLED_ATTRIBUTES)
        expected_unpickled_attrs.append("forward_async")
        expected_unpickled_attrs.append("forward")

        with self.assertRaisesRegex(
            RuntimeError, "Passing a script RemoteModule over RPC is not supported."
        ):
            # Create a remote module on worker1 and then pass it to worker2 over the RPC layer.
            for remote_module in self._create_remote_module_iter(
                dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
            ):
                # Test querying some simple attributes from worker2.
                rpc.rpc_sync(
                    dst_worker2_name, remote_module_attributes, (remote_module,)
                )

    @dist_utils.dist_init
    def test_create_remote_module_from_module_rref(self):
        # [FUNCTION DOC] Creates a remote module on worker1, builds another from its module RRef on
        # worker2 and checks that running forward on each gives equal results. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)

        # Create a remote module on worker1 and then pass its `module_rref` to worker2 over the RPC layer.
        for remote_module in self._create_remote_module_iter(
            dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            remote_module2 = rpc.rpc_sync(
                dst_worker2_name,
                RemoteModule.init_from_module_rref,
                (dst_worker2_name, remote_module.get_module_rref()),
            )

            args = (torch.ones(1), 2, "3")
            ret1 = rpc.rpc_sync(dst_worker1_name, remote_forward, (remote_module, args))
            ret2 = rpc.rpc_sync(
                dst_worker2_name, remote_forward, (remote_module2, args)
            )
            self.assertEqual(ret1, ret2)


class CudaRemoteModuleTest(CommonRemoteModuleTest):
    # [CLASS DOC] Tests of RemoteModule with CUDA devices on the destination worker; each test is
    # skipped when fewer than one GPU is available.
    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_valid_device(self):
        # [FUNCTION DOC] Checks that a remote module placed on cuda:0, addressed by worker name and
        # by rank, reports a cuda device with index 0. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = dist_utils.worker_name(dst_rank)

        for remote_module in self._create_remote_module_iter(
            f"{dst_worker_name}/cuda:0", modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            device = rpc.rpc_sync(
                dst_worker_name, remote_device, (remote_module.module_rref,)
            )
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device.index, 0)

        # Test rank works as well.
        for remote_module in self._create_remote_module_iter(
            f"rank:{dst_rank}/cuda:0", modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            device = rpc.rpc_sync(
                dst_worker_name, remote_device, (remote_module.module_rref,)
            )
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device.index, 0)

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_invalid_devices(self):
        # [FUNCTION DOC] Checks that malformed or invalid remote device strings (unknown type, bad
        # ordinal, bad cpu name, empty device, extra or missing parts) raise the expected
        # RuntimeError or ValueError with the expected message, with a platform specific message on
        # ROCm. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected one of .+ device type at start of device string",
        ):
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/foo",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        if TEST_WITH_ROCM:
            errorString = (
                r"HIP error: invalid device ordinal\n"
                r"HIP kernel errors might be asynchronously reported at some other API call, "
                r"so the stacktrace below might be incorrect.\n"
                r"For debugging consider passing AMD_SERIALIZE_KERNEL=3"
            )
        else:
            errorString = r"CUDA error: invalid device ordinal"
        with self.assertRaisesRegex(RuntimeError, errorString):
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/cuda:100",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        with self.assertRaisesRegex(RuntimeError, r"Invalid device string: 'cpu2'"):
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/cpu2",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        with self.assertRaisesRegex(RuntimeError, r"Device string must not be empty"):
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        with self.assertRaisesRegex(
            ValueError,
            r"Could not parse remote_device: worker1/cuda:0/cuda:1. The valid format is '<workername>/<device>'",
        ):
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    f"{dst_worker_name}/cuda:0/cuda:1",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        with self.assertRaisesRegex(
            ValueError,
            r"Could not parse remote_device: /. The valid format is '<workername>/<device>'",
        ):
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    "/",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

        with self.assertRaisesRegex(
            ValueError,
            r"Could not parse remote_device: /cuda:0. The valid format is '<workername>/<device>'",
        ):
            [
                m.forward()
                for m in self._create_remote_module_iter(
                    "/cuda:0",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            ]

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_input_moved_to_cuda_device(self):
        # [FUNCTION DOC] Checks that CPU tensor arguments (positional and keyword) are implicitly
        # moved to the cuda device and that results come back on CPU, for forward and
        # forward_async. Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # These two CPU tensors (in args and kwargs) should be implicitly moved to an appropriate cuda device.
        t1 = torch.ones(1)
        args = (t1, 2)
        t2 = t1 * 2
        kwargs = dict(word=t2)

        # Only test Python nn.Module, because script module methods don't support taking kwargs.
        for remote_module in self._create_remote_module_iter(
            f"{dst_worker_name}/cuda:0", modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            ret_fut = remote_module.forward_async(*args, **kwargs)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args + (t2,))))
            # TODO: Once the RPC backend can support directly sending GPU tensors, the expected device type should be "cuda:0".
            self.assertEqual(ret[0].device.type, "cpu")
            self.assertEqual(ret[2].device.type, "cpu")

            ret = remote_module.forward(*args, **kwargs)
            self.assertEqual(ret, tuple(reversed(args + (t2,))))
            # TODO: Once the RPC backend can support directly sending GPU tensors, the expected device type should be "cuda:0".
            self.assertEqual(ret[0].device.type, "cpu")
            self.assertEqual(ret[2].device.type, "cpu")

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_input_moved_to_cuda_device_script(self):
        # [FUNCTION DOC] Same as the CUDA input move test but through a scripted function on a
        # scripted remote module; checks the returned value and that the returned tensor is on CPU.
        # Runs on rank 0 only.
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        scripted_remote_module = next(
            self._create_remote_module_iter(
                f"{dst_worker_name}/cuda:0",
                modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE],
            )
        )

        @torch.jit.script
        def run_forward(scripted_remote_module: MyModuleInterface):
            # [FUNCTION DOC] HELPER: Scripted function that calls forward on the module and returns
            # the result.
            ret = scripted_remote_module.forward(torch.ones(1), 2, "3")
            return ret

        ret = run_forward(scripted_remote_module)

        self.assertEqual(ret, ("3", 2, torch.ones(1)))
        # TODO: Once the RPC backend can support directly sending GPU tensors, the expected device type should be "cuda:0".
        self.assertEqual(ret[2].device.type, "cpu")
