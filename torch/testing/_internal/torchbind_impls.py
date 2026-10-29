# mypy: allow-untyped-defs
# [MODULE DOC] Registers Python fake/meta implementations and fake classes for the
# _TorchScriptTesting torchbind test library (Foo, ContainsTensor, TensorQueue, ...). Main
# entry is init_torchbind_implementations, which loads the test library once and registers
# everything. Module globals track initialization and a shared empty tensor queue. No
# import-time side effects beyond defining the globals.
import contextlib
from pathlib import Path

import torch


_TORCHBIND_IMPLS_INITIALIZED = False

_TENSOR_QUEUE_GLOBAL_TEST: torch.ScriptObject | None = None


def init_torchbind_implementations():
    # [FUNCTION DOC] Idempotent one-time setup: loads the torchbind test library, registers
    # fake operators and fake classes, creates the global test tensor queue (see
    # _empty_tensor_queue) and sets the initialized flag. Mutates module globals and
    # global operator registries. Returns nothing; the load step raises SkipTest on macOS.
    global _TORCHBIND_IMPLS_INITIALIZED
    global _TENSOR_QUEUE_GLOBAL_TEST
    if _TORCHBIND_IMPLS_INITIALIZED:
        return

    load_torchbind_test_lib()
    register_fake_operators()
    register_fake_classes()
    _TENSOR_QUEUE_GLOBAL_TEST = _empty_tensor_queue()
    _TORCHBIND_IMPLS_INITIALIZED = True


def _empty_tensor_queue() -> torch.ScriptObject:
    # [FUNCTION DOC] HELPER: Constructs a TensorQueue script object from the registered
    # test class, initialized with an empty tensor filled with -1 (its init tensor).
    return torch.classes._TorchScriptTesting._TensorQueue(
        torch.empty(
            0,
        ).fill_(-1)
    )


# put these under a function because the corresponding library might not be loaded yet.
def register_fake_operators():
    # [FUNCTION DOC] Registers fake (abstract) implementations for the _TorchScriptTesting
    # operators takes_foo_python_meta, queue_pop, queue_push, queue_size and
    # takes_foo_tensor_return, autocast policies (float32 on cpu and cuda) for queue_push
    # and queue_pop, and Meta-key python kernels for takes_foo_list_return,
    # takes_foo_tuple_return and takes_foo. Requires the test library to be loaded; raises
    # if an operator is missing or already registered. Mutates global registries.
    @torch.library.register_fake("_TorchScriptTesting::takes_foo_python_meta")
    def fake_takes_foo(foo, z):
        # [FUNCTION DOC] HELPER: Fake impl; returns the foo object's add_tensor of z.
        return foo.add_tensor(z)

    @torch.library.register_fake("_TorchScriptTesting::queue_pop")
    def fake_queue_pop(tq):
        # [FUNCTION DOC] HELPER: Fake impl; pops and returns from the queue object.
        return tq.pop()

    @torch.library.register_fake("_TorchScriptTesting::queue_push")
    def fake_queue_push(tq, x):
        # [FUNCTION DOC] HELPER: Fake impl; pushes x onto the queue object.
        return tq.push(x)

    torch.library.register_autocast(
        "_TorchScriptTesting::queue_push", "cpu", torch.float32
    )
    torch.library.register_autocast(
        "_TorchScriptTesting::queue_push", "cuda", torch.float32
    )

    torch.library.register_autocast(
        "_TorchScriptTesting::queue_pop", "cpu", torch.float32
    )
    torch.library.register_autocast(
        "_TorchScriptTesting::queue_pop", "cuda", torch.float32
    )

    @torch.library.register_fake("_TorchScriptTesting::queue_size")
    def fake_queue_size(tq):
        # [FUNCTION DOC] HELPER: Fake impl; returns the queue object's size.
        return tq.size()

    def meta_takes_foo_list_return(foo, x):
        # [FUNCTION DOC] HELPER: Meta impl; applies add_tensor three times in a chain
        # starting from x and returns the three results as a list.
        a = foo.add_tensor(x)
        b = foo.add_tensor(a)
        c = foo.add_tensor(b)
        return [a, b, c]

    def meta_takes_foo_tuple_return(foo, x):
        # [FUNCTION DOC] HELPER: Meta impl; applies add_tensor twice in a chain starting
        # from x and returns the two results as a tuple.
        a = foo.add_tensor(x)
        b = foo.add_tensor(a)
        return (a, b)

    @torch.library.register_fake("_TorchScriptTesting::takes_foo_tensor_return")
    def meta_takes_foo_tensor_return(foo, x):
        # [FUNCTION DOC] HELPER: Fake impl; ignores foo and x and returns an int32 cpu
        # tensor of rank 2 whose two sizes are new unbacked dynamic sizes from the
        # library context.
        # This implementation deliberately creates unbacked symint for testing
        ctx = torch.library.get_ctx()
        fake_shape = [ctx.new_dynamic_size() for _ in range(2)]
        return torch.empty(fake_shape, dtype=torch.int, device="cpu")

    torch.ops._TorchScriptTesting.takes_foo_list_return.default.py_impl(
        torch._C.DispatchKey.Meta
    )(meta_takes_foo_list_return)

    torch.ops._TorchScriptTesting.takes_foo_tuple_return.default.py_impl(
        torch._C.DispatchKey.Meta
    )(meta_takes_foo_tuple_return)

    torch.ops._TorchScriptTesting.takes_foo.default.py_impl(torch._C.DispatchKey.Meta)(
        # make signature match original cpp implementation to support kwargs
        lambda foo, x: foo.add_tensor(x)
    )


def register_fake_classes():
    # [FUNCTION DOC] Registers Python fake classes (FakeFoo, FakeContainsTensor,
    # FakeTensorQueue, FakeFlatten) for the corresponding _TorchScriptTesting script
    # classes via the fake class registry. Raises if a class is already registered or the
    # library is not loaded. Mutates global registries; returns nothing.
    @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
    class FakeFoo:
        # [CLASS DOC] HELPER: Fake of the Foo script class holding ints x and y.
        def __init__(self, x: int, y: int):
            # [FUNCTION DOC] HELPER: Stores x and y.
            self.x = x
            self.y = y

        @classmethod
        def __obj_unflatten__(cls, flattend_foo):
            # [FUNCTION DOC] HELPER: Builds an instance from a flattened state, a sequence
            # of (name, value) pairs converted to keyword arguments.
            return cls(**dict(flattend_foo))

        def add_tensor(self, z):
            # [FUNCTION DOC] HELPER: Returns z scaled by the sum of x and y.
            return (self.x + self.y) * z

    @torch._library.register_fake_class("_TorchScriptTesting::_ContainsTensor")
    class FakeContainsTensor:
        # [CLASS DOC] HELPER: Fake of ContainsTensor holding one tensor t.
        def __init__(self, t: torch.Tensor):
            # [FUNCTION DOC] HELPER: Stores the tensor t.
            self.t = t

        @classmethod
        def __obj_unflatten__(cls, flattend_foo):
            # [FUNCTION DOC] HELPER: Builds an instance from (name, value) pairs passed as
            # keyword arguments.
            return cls(**dict(flattend_foo))

        def get(self):
            # [FUNCTION DOC] HELPER: Returns the stored tensor.
            return self.t

    @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
    class FakeTensorQueue:
        # [CLASS DOC] HELPER: Fake of TensorQueue; attribute queue is a Python list of
        # tensors used as a FIFO (push appends, pop removes from the front).
        def __init__(self, queue):
            # [FUNCTION DOC] HELPER: Stores the list of tensors as the queue.
            self.queue = queue

        @classmethod
        def __obj_unflatten__(cls, flattened_ctx):
            # [FUNCTION DOC] HELPER: Builds an instance from (name, value) pairs passed as
            # keyword arguments.
            return cls(**dict(flattened_ctx))

        def push(self, x):
            # [FUNCTION DOC] HELPER: Appends tensor x to the back of the queue.
            self.queue.append(x)

        def pop(self):
            # [FUNCTION DOC] HELPER: Removes and returns the front tensor; if the queue is
            # empty returns a new empty scalar-shaped uninitialized tensor instead.
            if self.is_empty():
                return torch.empty([])
            return self.queue.pop(0)

        def size(self):
            # [FUNCTION DOC] HELPER: Returns the number of queued tensors.
            return len(self.queue)

        def is_empty(self):
            # [FUNCTION DOC] HELPER: Returns True if the queue holds no tensors.
            return len(self.queue) == 0

        def float_size(self):
            # [FUNCTION DOC] HELPER: Returns the number of queued tensors as a float.
            return float(len(self.queue))

    @torch._library.register_fake_class("_TorchScriptTesting::_FlattenWithTensorOp")
    class FakeFlatten:
        # [CLASS DOC] HELPER: Fake of FlattenWithTensorOp holding one value t.
        def __init__(self, t):
            # [FUNCTION DOC] HELPER: Stores t.
            self.t = t

        def get(self):
            # [FUNCTION DOC] HELPER: Returns the stored value.
            return self.t

        @classmethod
        def __obj_unflatten__(cls, flattened_ctx):
            # [FUNCTION DOC] HELPER: Builds an instance from (name, value) pairs passed as
            # keyword arguments.
            return cls(**dict(flattened_ctx))


def load_torchbind_test_lib():
    # [FUNCTION DOC] Locates and loads the torchbind test shared library through the torch
    # ops library loader. Path depends on platform: internal build target path in fbcode or
    # sandcastle, torchbind_test.dll on Windows, libtorchbind_test.so elsewhere. Raises
    # unittest.SkipTest on macOS; other errors come from the lookup or loader.
    import unittest

    from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
        find_library_location,
        IS_FBCODE,
        IS_MACOS,
        IS_SANDCASTLE,
        IS_WINDOWS,
    )

    if IS_MACOS:
        raise unittest.SkipTest("non-portable load_library call used in test")
    elif IS_SANDCASTLE or IS_FBCODE:
        lib_file_path = Path("//caffe2/test/cpp/jit:test_custom_class_registrations")
    elif IS_WINDOWS:
        lib_file_path = find_library_location("torchbind_test.dll")
    else:
        lib_file_path = find_library_location("libtorchbind_test.so")
    torch.ops.load_library(str(lib_file_path))


@contextlib.contextmanager
def _register_py_impl_temporarily(op_overload, key, fn):
    # [FUNCTION DOC] HELPER: Context manager that registers fn as the python kernel for the
    # dispatch key on the operator overload, yields, then removes that kernel entry and
    # clears the overload's dispatch cache.
    # KNOWN ISSUE: the cleanup runs even if registration failed, so a failed registration
    # makes the cleanup raise a missing-key error that masks the original one.
    try:
        op_overload.py_impl(key)(fn)
        yield
    finally:
        del op_overload.py_kernels[key]
        op_overload._dispatch_cache.clear()
