# Owner(s): ["module: __torch_dispatch__"]
# ruff: noqa: F841
# [MODULE DOC] Tests for __torch_dispatch__ and the Python dispatcher/library registration
# machinery. Covers the dispatcher Python bindings, torch.library fallbacks and kernel overrides
# (TestPythonRegistration), tensor-subclass dispatch behavior using LoggingTensor and TwoTensor
# (TestPythonDispatch), plus mode, tracing, FakeTensor and sizes/strides tests further down.
# Import-time side effects: defines helper classes and instantiates parametrized tests and
# device-type tests; the file runs run_tests when executed as a script.

import gc
import pickle
import sys
import tempfile
import unittest
import weakref
from copy import deepcopy

import torch
import torch._dynamo
from torch import SymInt
from torch._C import DispatchKey, DispatchKeySet
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda.jiterator import _create_jit_fn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.library import _scoped_library, fallthrough_kernel, impl, Library
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    first_sample,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.logging_tensor import (
    capture_logs,
    capture_logs_with_logging_tensor_mode,
    log_input,
    LoggingTensor,
    LoggingTensorMode,
    LoggingTensorReentrant,
)
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import all_same_mode, no_dispatch
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    is_in_torch_dispatch_mode,
    is_traceable_wrapper_subclass,
    is_traceable_wrapper_subclass_type,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map, tree_map_only


# used as DataLoader collate_fn below; named here to avoid trying to pickle a lambda
def _identity(x):
    # [FUNCTION DOC] HELPER: Module-level identity function returning its single argument
    # unchanged. It exists only so a DataLoader collate function can be pickled by name (a
    # lambda cannot be).
    return x


class _TraceableWrapperSubclassTestBase(torch.Tensor):
    # [CLASS DOC] HELPER: Minimal wrapper tensor subclass base used by the
    # traceable-wrapper-subclass protocol test. It stores one inner tensor in the slot named
    # elem and mirrors its size, stride, dtype, layout, device, requires_grad and storage
    # offset. Any dispatched op raises, so instances are only usable for protocol/metadata
    # checks.
    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem):
        # [FUNCTION DOC] HELPER: Builds a wrapper subclass instance via the private
        # make-wrapper-subclass hook, copying all metadata from elem, then stores elem in the
        # elem slot and returns the wrapper. No storage is allocated for the wrapper.
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            elem.size(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        )
        r.elem = elem
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch hook that always raises RuntimeError with message NYI,
        # so any actual op on the wrapper fails.
        raise RuntimeError("NYI")


class _StaticUnflattenWrapper(_TraceableWrapperSubclassTestBase):
    # [CLASS DOC] HELPER: Traceable wrapper subclass whose unflatten hook is a staticmethod; the
    # flatten hook reports the inner tensor name list (elem) and a metadata mapping tagged as
    # static.
    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Returns the pair of inner attribute names (just elem) and a
        # metadata mapping with kind set to static.
        return ["elem"], {"kind": "static"}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Static unflatten hook: ignores metadata, outer size and stride
        # and rebuilds a _StaticUnflattenWrapper from the elem entry of the inner tensors
        # mapping.
        return _StaticUnflattenWrapper(inner_tensors["elem"])


class _ClassmethodUnflattenWrapper(_TraceableWrapperSubclassTestBase):
    # [CLASS DOC] HELPER: Same as the static variant but its unflatten hook is a classmethod,
    # used to verify the runtime protocol check accepts both forms.
    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Returns the pair of inner attribute names (just elem) and a
        # metadata mapping with kind set to classmethod.
        return ["elem"], {"kind": "classmethod"}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Classmethod unflatten hook: rebuilds an instance of cls from
        # the elem entry of the inner tensors mapping, ignoring metadata, size and stride.
        return cls(inner_tensors["elem"])


class TestDispatcherPythonBindings(TestCase):
    # [CLASS DOC] Test case for the low-level dispatcher Python bindings (private _dispatch_*
    # functions on torch._C). Currently holds a single boxed-call test.
    def test_call_boxed(self) -> None:
        # [FUNCTION DOC] Looks up the schema for aten sin through the private dispatcher
        # binding, calls it boxed with a random tensor, and checks the result equals the regular
        # sin method.
        sin = torch._C._dispatch_find_schema_or_throw("aten::sin", "")
        x = torch.randn(3)
        y = torch._C._dispatch_call_boxed(sin, x)
        self.assertEqual(y, x.sin())


class TestPythonRegistration(TestCase):
    # [CLASS DOC] Tests for registering kernels and fallbacks from Python via torch.library
    # (Library, _scoped_library, impl, define). Class attribute test_ns names the scratch
    # operator namespace; tearDown removes that namespace from torch.ops. Covers fallbacks on
    # test dispatch keys, TLS key set guards, kernel override and restoration, Library
    # finalizer/teardown behavior and error cases.
    test_ns = "_test_python_registration"

    def tearDown(self):
        # [FUNCTION DOC] Removes the scratch test namespace attribute from torch.ops if it
        # exists, so ops cached by one test do not leak into the next.
        if hasattr(torch.ops, self.test_ns):
            del torch.ops._test_python_registration

    def test_global_enter(self):
        # [FUNCTION DOC] Regression test: enters a LoggingTensorMode manually, drops the only
        # Python reference so the C++ mode stack is the sole owner, then runs a tensor op and
        # checks it does not crash. The mode is exited (via a weak reference) in a finally
        # block.
        try:
            v = LoggingTensorMode()
            v_ref = weakref.ref(v)

            v.__enter__()
            # The bug trigger when the C++ stack is the only
            # owner of the mode object.
            del v

            # Does not segfault
            str(torch.rand(2))

        finally:
            v_ref().__exit__(None, None, None)

    def test_fallback(self) -> None:
        # [FUNCTION DOC] Registers a Python fallback on a testing-only dispatch key and enables
        # that key through a forced TLS guard. Checks the fallback receives the expected op,
        # args and kwargs for a factory op (empty) and a regular op (add), and that the tensor
        # returned by the fallback (with an arbitrary shape) is what the caller sees.
        test_key = "TESTING_ONLY_GenericMode"
        test_keyset = torch._C.DispatchKeySet(test_key)
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            expected_op = None
            expected_args = None
            expected_kwargs = None
            # Use this out shape to make sure the result from our fallback
            # is what is returned to the user
            out_shape = None

            def my_fallback(op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Fallback kernel. With the test key excluded from
                # dispatch, it asserts the op, args and kwargs match the values the test
                # currently expects, then returns an empty tensor of the shape chosen by the
                # test.
                # Disable our handler during checks and generating the output
                with torch._C._ForceDispatchKeyGuard(
                    include_to_set, exclude_to_set | test_keyset
                ):
                    self.assertIs(op, expected_op)
                    self.assertEqual(args, expected_args)
                    self.assertEqual(kwargs, expected_kwargs)
                    # Return something specific
                    return torch.empty(out_shape)

            my_lib.fallback(my_fallback, test_key)

            a, b = torch.rand(2), torch.rand(2)

            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                # Check a factory function
                expected_op = torch.ops.aten.empty.memory_format
                expected_args = ((2, 2),)
                # Extra kwargs to bypass issues with default args in factory functions
                expected_kwargs = {
                    "dtype": torch.float64,
                    "pin_memory": False,
                    "device": torch.device("cpu"),
                }
                out_shape = (3,)
                out = torch.empty(*expected_args, **expected_kwargs)
                self.assertEqual(out.size(), out_shape)

                # Check a regular function
                expected_op = torch.ops.aten.add.Tensor
                expected_args = (a, b)
                expected_kwargs = {}
                out_shape = (4,)
                out = a + b
                self.assertEqual(out.size(), out_shape)

    def test_fallback_keyset(self) -> None:
        # [FUNCTION DOC] Registers two fallbacks on two testing-only keys, the first with keyset
        # access. Verifies the second fallback (which turns an add into a subtraction) runs,
        # that its nested subtraction re-enters the first fallback, and that the final result
        # equals subtraction.
        test_key_first = "TESTING_ONLY_GenericMode"
        test_key_second = "TESTING_ONLY_GenericWrapper"
        test_keyset = torch._C.DispatchKeySet(test_key_first) | torch._C.DispatchKeySet(
            test_key_second
        )
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            first_called = False
            second_called = False

            def first_fallback(keyset, op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: First fallback, taking the key set. If the second
                # fallback already ran, it marks itself called and re-runs the op with the test
                # keys excluded; otherwise it removes its own key from the key set and
                # redispatches the op below it.
                nonlocal first_called
                if second_called:
                    # Recursive call
                    first_called = True
                    with torch._C._ForceDispatchKeyGuard(
                        include_to_set, exclude_to_set | test_keyset
                    ):
                        return op(*args, **kwargs)
                else:
                    # Redispatch down
                    keyset = keyset.remove(test_key_first)
                    return op.redispatch(keyset, *args, **kwargs)

            def second_fallback(op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Second fallback. Marks itself called, asserts the first
                # fallback has not yet run, computes a subtraction of the two args instead of
                # the requested op (which re-enters the first fallback), asserts the first has
                # now run, and returns the difference.
                nonlocal second_called
                # Set to avoid infinite recursion
                second_called = True
                # New dispatcher call should hit the first callback again
                self.assertFalse(first_called)
                a, b = args
                # Make a subtraction here instead of add !
                c = a - b
                self.assertTrue(first_called)
                return c

            my_lib.fallback(first_fallback, test_key_first, with_keyset=True)
            my_lib.fallback(second_fallback, test_key_second)

            a, b = torch.rand(2), torch.rand(2)
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                c = a + b

            self.assertEqual(c, a - b)
            self.assertTrue(first_called)
            self.assertTrue(second_called)

    def test_fallback_fallthrough(self) -> None:
        # [FUNCTION DOC] Registers a fallthrough kernel on one testing-only key and a recording
        # fallback on another. Adds two tensors with both keys enabled and checks the result
        # equals normal addition and that the recording fallback was invoked.
        test_key_first = "TESTING_ONLY_GenericMode"
        test_key_second = "TESTING_ONLY_GenericWrapper"
        test_keyset = torch._C.DispatchKeySet(test_key_first) | torch._C.DispatchKeySet(
            test_key_second
        )
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            is_called = False

            def my_fallback(op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Fallback that records it was called and re-runs the op
                # with the test keys excluded, returning the real result.
                nonlocal is_called
                is_called = True
                with torch._C._ForceDispatchKeyGuard(
                    include_to_set, exclude_to_set | test_keyset
                ):
                    return op(*args, **kwargs)

            my_lib.fallback(torch.library.fallthrough_kernel, test_key_first)
            my_lib.fallback(my_fallback, test_key_second)

            a, b = torch.rand(2), torch.rand(2)
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                c = a + b

            self.assertEqual(c, a + b)
            self.assertTrue(is_called)

    def test_fallthrough_for_dense_key_with_meta_in_tls(self) -> None:
        # [FUNCTION DOC] Defines a custom sum op with a fallthrough CPU kernel and a Meta
        # kernel. With the Meta key forced into the TLS include set, calling the op on a CPU
        # tensor must hit the Meta kernel, proving the fallthrough on the dense key does not
        # shadow it.
        # This tests that if meta is included in TlS dispatch key set,
        # then a meta kernel should be called regardless if a dense
        # backend has a fallthrough kernel

        a = torch.randn((3, 3))
        with _scoped_library("custom", "DEF") as my_lib:
            my_lib.define("sum(Tensor self) -> Tensor")
            meta_is_called = False

            def sum_meta(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Meta kernel that records it ran (through the enclosing
                # flag) and returns its first argument.
                nonlocal meta_is_called
                meta_is_called = True
                return args[0]

            my_lib.impl("sum", fallthrough_kernel, "CPU")
            my_lib.impl("sum", sum_meta, "Meta")

            with torch._C._IncludeDispatchKeyGuard(torch.DispatchKey.Meta):
                torch.ops.custom.sum.default(a)
                self.assertTrue(meta_is_called)

    def test_include_dispatch_key_guard_restores_tls_exactly(self) -> None:
        # [FUNCTION DOC] Checks that entering and leaving the include-dispatch-key guard for the
        # Meta key leaves the TLS local include set exactly as it was, by comparing raw
        # representations before and after.
        before = torch._C._dispatch_tls_local_include_set().raw_repr()
        with torch._C._IncludeDispatchKeyGuard(torch.DispatchKey.Meta):
            pass
        after = torch._C._dispatch_tls_local_include_set().raw_repr()
        self.assertEqual(before, after)

    @parametrize(
        "key",
        [
            torch.DispatchKey.Meta,
            torch.DispatchKey.CUDA,
            torch.DispatchKey.CPU,
        ],
    )
    def test_exclude_dispatch_key_guard_restores_tls_exactly(self, key) -> None:
        # [FUNCTION DOC] Parametrized over the Meta, CUDA and CPU keys. Checks that entering and
        # leaving the exclude-dispatch-key guard for a key set made from that key restores the
        # TLS local exclude set exactly, comparing raw representations.
        keyset = torch._C.DispatchKeySet(key)
        before = torch._C._dispatch_tls_local_exclude_set().raw_repr()
        with torch._C._ExcludeDispatchKeyGuard(keyset):
            pass
        after = torch._C._dispatch_tls_local_exclude_set().raw_repr()
        self.assertEqual(before, after)

    def test_dispatchkeyset_pickle(self) -> None:
        # [FUNCTION DOC] Round-trips a DispatchKeySet through pickle dumps and loads and checks
        # the loaded set equals the original.
        keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        serialized = pickle.dumps(keyset)
        new_keyset = pickle.loads(serialized)
        self.assertEqual(new_keyset, keyset)

    def test_dispatchkeyset_eq(self) -> None:
        # [FUNCTION DOC] Checks DispatchKeySet equality and inequality operators: two sets built
        # from the same key compare equal and not unequal, while sets from different keys
        # compare unequal.
        a = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        b = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        c = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertTrue(a != c)

    def test_override_aten_ops_with_multiple_libraries(self) -> None:
        # [FUNCTION DOC] Overrides aten neg (as a view producing implementation on the autograd
        # CPU key) and aten mul (on the ZeroTensor key) from two overlapping scoped libraries.
        # Verifies the overrides take effect, that a mismatched namespace in impl raises
        # RuntimeError, that re-registering the same op/key without allowing override raises
        # RuntimeError, that removing the inner library does not affect the outer, and that
        # original behavior returns once both libraries exit.
        x = torch.tensor([1, 2])
        with _scoped_library("aten", "IMPL") as my_lib2:
            with _scoped_library("aten", "IMPL") as my_lib1:
                # Example 1
                def my_neg(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: Replacement neg kernel returning the negative view
                    # of its first argument.
                    return args[0]._neg_view()

                # Now we are secretly making the operator a view op so autograd needs to know how
                # to handle it
                my_lib1.impl("neg", my_neg, "AutogradCPU")

                self.assertTrue(torch.neg(x).is_neg())

                # RuntimeError: impl("aten::neg", ...):
                # Explicitly provided namespace (aten) in operator name does not match ...
                with self.assertRaisesRegex(
                    RuntimeError, "operator name does not match namespace"
                ):
                    with _scoped_library("foo", "DEF") as my_lib3:
                        my_lib3.define("neg(Tensor self) -> Tensor")
                        my_lib3.impl(torch.ops.aten.neg.default, my_neg, "AutogradCPU")

                # Example 2
                def my_mul(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: Replacement mul kernel returning a zeros tensor
                    # shaped like its first argument.
                    return torch.zeros_like(args[0])

                # torch.ops.aten.mul.Tensor
                my_lib2.impl("aten::mul.Tensor", my_mul, "ZeroTensor")

                y = torch._efficientzerotensor(2)
                self.assertFalse(torch.mul(x, y)._is_zerotensor())

                # Assert that a user can't override the behavior of a (ns, op, dispatch_key)
                # combination if someone overridden the behavior for the same before them
                with self.assertRaisesRegex(
                    RuntimeError, "already a kernel registered from python"
                ):
                    my_lib2.impl(
                        torch.ops.aten.mul.Tensor,
                        my_mul,
                        "ZeroTensor",
                        allow_override=False,
                    )

            # Validate that lib2 is not affected by removing lib1
            self.assertFalse(torch.mul(x, y)._is_zerotensor())

        # Validate that the old behavior is restored for neg and mul
        self.assertFalse(torch.neg(x).is_neg())
        self.assertTrue(torch.mul(x, y)._is_zerotensor())

    def test_error_if_fn_not_callable(self):
        # [FUNCTION DOC] Checks that registering a non-callable object (an empty list) as an
        # impl raises TypeError mentioning that the input function must be callable.
        with self.assertRaisesRegex(
            TypeError, "Input function is required to be a callable"
        ):
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl(torch.ops.aten.neg.default, [], "AutogradCPU")

    def test_finalizer(self):
        # [FUNCTION DOC] Verifies Library finalizer reference counting. After defining an op and
        # registering a CPU impl, checks reference counts of the library, the global impls
        # registry and the library's own impl table (adjusting expectations for Python 3.14),
        # checks the registry key exists, then deletes the library and checks the key is removed
        # and the registry refcount returns to its starting value.
        impls_refcnt = sys.getrefcount(torch.library._impls)
        lib = Library(self.test_ns, "FRAGMENT")  # noqa: SCOPED_LIBRARY
        lib.define("foo123(Tensor x) -> Tensor")

        # 1 for `lib`, 1 for sys.getrefcount' for previous python version (<=3.12)
        # In Python 3.13+, sys.getrefcount() was optimized to not create
        # a temporary reference, so expected counts are 1 less than before
        expected_refcount = 1 if sys.version_info >= (3, 14) else 2
        self.assertEqual(sys.getrefcount(lib), expected_refcount)

        # We gained an additional reference that gets cleared when the finalizer runs
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt + 1)
        # 1 for `lib`
        # 1 for the finalizer
        # 1 for sys.getrefcount
        self.assertEqual(sys.getrefcount(lib._op_impls), 3)

        def foo123(x):
            # [FUNCTION DOC] HELPER: Empty kernel registered only to create a registry entry.
            pass

        lib.impl(f"{self.test_ns}::foo123", foo123, "CPU")
        key = f"{self.test_ns}/foo123/CPU"
        self.assertTrue(key in torch.library._impls)

        saved_op_impls = lib._op_impls

        # del will definitely work if the following passes
        self.assertEqual(sys.getrefcount(lib), expected_refcount)
        del lib

        # 1 for saved_op_impls
        # 1 for sys.getrefcount
        # This function should be the last user of lib._op_impls:
        # - lib should not have a reference anymore (it was del'ed)
        # - lib's finalizer should not have a reference anymore
        self.assertEqual(sys.getrefcount(saved_op_impls), expected_refcount)

        self.assertTrue(key not in torch.library._impls)

        # lib's finalizer should not have a reference anymore
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt)

    @skipIfTorchDynamo(
        "dynamo's tracing keeps a reference to `lib`, so `del lib` doesn't "
        "trigger the finalizer; this test is exclusively about gc-driven "
        "Library teardown."
    )
    def test_finalizer_clears_torch_ops_cache(self):
        # [FUNCTION DOC] Regression test (skipped under Dynamo): defines and registers an op,
        # calls it so the packet is cached on the torch.ops namespace, deletes the library and
        # collects garbage, then checks the op name is gone from the namespace and that the
        # export utility collecting valid CIA ops for the namespace does not raise.
        # Regression test for https://github.com/pytorch/pytorch/issues/181765:
        # `del lib` (running the finalizer) must clear the cached
        # OpOverloadPacket on torch.ops.<ns>, otherwise iterating ops in that
        # namespace later raises AttributeError.
        lib = Library(self.test_ns, "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_func() -> None")

        @impl(lib, "my_func", "")
        def my_func():
            # [FUNCTION DOC] HELPER: Empty kernel for the defined op, registered via the impl
            # decorator.
            pass

        torch.ops._test_python_registration.my_func()
        del lib
        gc.collect()

        namespace = torch.ops._test_python_registration
        self.assertNotIn("my_func", namespace._dir)
        from torch._export.utils import _collect_all_valid_cia_ops_for_namespace

        _collect_all_valid_cia_ops_for_namespace(namespace)

    def test_finalizer_no_getattr_into_cpp(self):
        # [FUNCTION DOC] Regression test: defines and registers an op but never accesses it
        # through torch.ops. Temporarily replaces the namespace class getattr with a recording
        # wrapper, deletes the library and collects garbage, restores the original, then asserts
        # the op name was never looked up (the cache-clearing finalizer must not call into C++).
        # Regression test: _clear_torch_ops_cache must not call __getattr__ on
        # _OpNamespace (which invokes _jit_get_operation in C++). During
        # interpreter shutdown the C++ runtime may be torn down, causing
        # UnicodeDecodeError or segfaults when the finalizer fires.
        lib = Library(self.test_ns, "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("uncached_op() -> None")

        @impl(lib, "uncached_op", "")
        def uncached_op():
            # [FUNCTION DOC] HELPER: Empty kernel for the defined op.
            pass

        # Do NOT call torch.ops._test_python_registration.uncached_op() here,
        # so the OpOverloadPacket is never cached on the namespace object.
        # This simulates what happens when a library (e.g. vLLM) registers ops
        # but some are never looked up before shutdown.
        namespace = torch.ops._test_python_registration

        # Patch __getattr__ to detect if it gets called during cache clearing.
        original_getattr = type(namespace).__getattr__
        getattr_called_with = []

        def tracking_getattr(self, name):
            # [FUNCTION DOC] HELPER: Wrapper getattr that appends the requested name to a list
            # and delegates to the original getattr.
            getattr_called_with.append(name)
            return original_getattr(self, name)

        type(namespace).__getattr__ = tracking_getattr
        try:
            del lib
            gc.collect()
        finally:
            type(namespace).__getattr__ = original_getattr

        self.assertNotIn(
            "uncached_op",
            getattr_called_with,
            "_clear_torch_ops_cache must not trigger __getattr__ (which calls "
            "into C++) for ops that were never cached on the namespace",
        )

    def test_define_normalizes_namespaced_schema_name(self):
        # [FUNCTION DOC] Regression test: defines an op with a schema name prefixed by the same
        # namespace and checks the recorded op-definition name contains the namespace once (not
        # duplicated), then calls the library destroy method, which must not raise.
        # Library.define() takes a bare op name in the schema, but C++ also
        # accepts a name prefixed with the matching namespace ("ns::foo"). In
        # that case the qualname recorded in _op_defs must not double-prepend the
        # namespace. Regression: it used to produce "ns::ns::foo", which later
        # broke teardown's _clear_torch_ops_cache with "too many values to
        # unpack (expected 2)".
        lib = Library(self.test_ns, "DEF")  # noqa: SCOPED_LIBRARY
        lib.define(f"{self.test_ns}::my_op(Tensor x) -> Tensor")
        self.assertIn(f"{self.test_ns}::my_op", lib._op_defs)
        self.assertNotIn(f"{self.test_ns}::{self.test_ns}::my_op", lib._op_defs)
        # Teardown must not raise.
        lib._destroy()

    def test_clear_torch_ops_cache_tolerates_malformed_qualname(self):
        # [FUNCTION DOC] Calls the cache-clearing helper from torch.library with malformed
        # qualified names (too many separators, and no separator); it must silently tolerate
        # them and not raise.
        # Defense-in-depth: _clear_torch_ops_cache runs in a shutdown-time
        # finalizer and must never raise, even on a qualname that does not split
        # into exactly "namespace::name".
        from torch.library import _clear_torch_ops_cache

        # Must not raise.
        _clear_torch_ops_cache({"a::b::c", "no_separator"})

    def test_register_check_mem_op_survives_gc(self):
        # [FUNCTION DOC] Regression guard: clears the cache of the inductor
        # register_check_mem_op helper, calls it, collects garbage, and checks the
        # _inductor_debug namespace and its check_memory_step op remain visible on torch.ops,
        # i.e. the helper keeps its Library alive.
        # Regression guard for the autorevert of #181785: inductor's
        # `register_check_mem_op` is an lru_cache-decorated registration whose
        # `lib` is a local variable. Once the finalizer started clearing the
        # torch.ops cache (this PR), the op vanished from torch.ops as soon as
        # the registration function returned, so later access to
        # `torch.ops._inductor_debug.check_memory_step.default` raised
        # AttributeError. The fix is in debug_utils.py (keep `lib` alive); this
        # test guards that the contract holds end-to-end.
        from torch._inductor.runtime import debug_utils

        debug_utils.register_check_mem_op.cache_clear()
        debug_utils.register_check_mem_op()
        gc.collect()

        self.assertTrue(hasattr(torch.ops, "_inductor_debug"))
        self.assertTrue(
            hasattr(torch.ops._inductor_debug, "check_memory_step"),
            "register_check_mem_op must keep its Library alive so the op "
            "remains in torch.ops after the finalizer would otherwise fire.",
        )

    def test_override_cpu_sum(self) -> None:
        # [FUNCTION DOC] Overrides aten sum on the CPU key with a kernel that records it ran and
        # returns a clone of its input. Checks the override is used inside the scoped library
        # and the true sum (3) is restored after.
        # Example 1
        run = [False]

        def my_sum(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement sum kernel that sets a flag in the enclosing
            # list and returns a clone of its first argument.
            run[0] = True
            return args[0].clone()

        with _scoped_library("aten", "IMPL") as my_lib1:
            my_lib1.impl("aten::sum", my_sum, "CPU")
            x = torch.tensor([1, 2])
            self.assertEqual(torch.sum(x), x)
            self.assertTrue(run[0])
        # Validate that the old behavior is restored for sum
        self.assertEqual(torch.sum(x), torch.tensor(3))

    def test_override_cuda_with_jiterator(self) -> None:
        # [FUNCTION DOC] Defines four CUDA-only scenarios, run only when CUDA is available and
        # not ROCm. Each overrides an aten kernel with a Jiterator-compiled kernel and checks
        # both the override result and restoration after the scoped library exits: inverted
        # where, relu as gelu, clipped exp for float16, and add that is off by one. On other
        # setups the test does nothing.
        def override_where_cuda() -> None:
            # [FUNCTION DOC] HELPER: Scenario overriding the CUDA where kernel with a Jiterator
            # kernel that inverts the condition; checks the swapped result, that the override
            # was called, and that normal behavior returns afterward.
            # Example 1: Invert the behavior of where's condition input
            not_where_code_string = """
            template <typename T> T inverted_where(bool cond, T a, T b){
                return !cond ? a : b;
            }
            """
            jitted_where = _create_jit_fn(not_where_code_string)

            CALLED = [False]

            def inverted_where(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper that sets a called flag and forwards to the
                # jitted inverted-where kernel.
                CALLED[0] = True
                return jitted_where(*args, **kwargs)

            # overriding where's cuda kernel with Jiterator generated kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::where.self", inverted_where, "CUDA")

                device = "cuda"
                cond = torch.tensor(
                    [True, True, False], device=device, dtype=torch.bool
                )
                x = torch.tensor([1, 2, 3], device=device)
                y = torch.tensor([-1, -2, -3], device=device)

                self.assertEqual(torch.where(cond, x, y), torch.tensor([-1, -2, 3]))
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(torch.where(cond, x, y), torch.tensor([1, 2, -3]))

        def override_gelu_cuda() -> None:
            # [FUNCTION DOC] HELPER: Scenario overriding CUDA gelu with a Jiterator kernel
            # computing relu; checks gelu equals relu while overridden and differs after
            # restoration.
            # Example 2: Use relu to approximate gelu for faster compute
            fastest_gelu_code_string = """
            template <typename T> T fast_gelu(T a){
                return a > 0 ? a : 0;
            }
            """
            jitted_gelu = _create_jit_fn(fastest_gelu_code_string)

            CALLED = [False]

            def fast_gelu(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper that sets a called flag and forwards to the
                # jitted relu-as-gelu kernel.
                CALLED[0] = True
                return jitted_gelu(*args, **kwargs)

            # overriding gelu's cuda kernel with Jiterator generated relu kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::gelu", fast_gelu, "CUDA")

                x = torch.rand([3, 3], device="cuda", dtype=torch.float)
                self.assertEqual(
                    torch.nn.functional.gelu(x), torch.nn.functional.relu(x)
                )
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertNotEqual(
                torch.nn.functional.gelu(x), torch.nn.functional.relu(x)
            )

        def override_exp_cuda() -> None:
            # [FUNCTION DOC] HELPER: Scenario overriding CUDA exp with a Jiterator kernel that
            # clamps large inputs to a fixed value; checks float16 results while overridden and
            # that exp of 100 overflows to infinity once restored.
            # Example 3: Preventing exp from exploding for float16
            clipped_exp_code_string = """
            template <typename T> T clipped_exp(T a){
                return a > T(10.0) ? T(22026.4657948) : exp(a);
            }
            """
            jitted_exp = _create_jit_fn(clipped_exp_code_string)

            CALLED = [False]

            def clipped_exp(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper that sets a called flag and forwards to the
                # jitted clipped-exp kernel.
                CALLED[0] = True
                return jitted_exp(*args, **kwargs)

            # overriding exp's cuda kernel with clipped_exp kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::exp", clipped_exp, "CUDA")

                x = torch.tensor([0.0, 100.0], device="cuda", dtype=torch.float16)
                self.assertEqual(
                    torch.exp(x),
                    torch.tensor([1.0, 22026.4657948], dtype=torch.float16),
                )
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(
                torch.exp(x), torch.tensor([1.0, torch.inf], dtype=torch.float16)
            )

        def override_add_cuda() -> None:
            # [FUNCTION DOC] HELPER: Scenario overriding CUDA add with a Jiterator kernel that
            # adds one extra; compares CUDA sums against CPU sums plus one while overridden and
            # plain CPU sums after restoration.
            # Example 4: simulate a hardware bug, where the adder is always off by 1
            buggy_add_code_string = """
            template <typename T> T buggy_add(T a, T b){
                return a + b + T(1);
            }
            """
            jitted_add = _create_jit_fn(buggy_add_code_string)

            CALLED = [False]

            def buggy_add(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper that sets a called flag and forwards to the
                # jitted buggy-add kernel.
                CALLED[0] = True
                return jitted_add(*args, **kwargs)

            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::add.Tensor", buggy_add, "CUDA")

                x_cpu = torch.rand([3, 3], device="cpu")
                y_cpu = torch.rand([3], device="cpu")

                x_cuda = x_cpu.cuda()
                y_cuda = y_cpu.cuda()

                self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu + 1)
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu)

        if torch.cuda.is_available() and not TEST_WITH_ROCM:
            override_where_cuda()
            override_gelu_cuda()
            override_exp_cuda()
            override_add_cuda()

    def test_extend_library_with_dispatch_key_arg(self):
        # [FUNCTION DOC] Checks that in a library created with a CPU dispatch key argument, impl
        # with a conflicting key (Conjugate) raises RuntimeError about inconsistent dispatch
        # key, while impl of aten sum without an explicit key works and the override gives the
        # expected result.
        def my_sum(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Sum kernel returning a clone of its first argument.
            return args[0].clone()

        with _scoped_library("aten", "IMPL", dispatch_key="CPU") as my_lib1:
            # RuntimeError: Explicitly provided dispatch key (Conjugate) is
            # inconsistent with the dispatch key of the enclosing TORCH_LIBRARY_IMPL block
            with self.assertRaisesRegex(
                RuntimeError, "inconsistent with the dispatch key"
            ):
                my_lib1.impl("sum", my_sum, "Conjugate")
            my_lib1.impl("aten::sum", my_sum)
            x = torch.tensor([1, 2])
            self.assertEqual(torch.sum(x), x)

    def test_create_new_library(self) -> None:
        # [FUNCTION DOC] Creates a new DEF library with a custom sum op, registers a CPU kernel
        # via the impl decorator and checks the op returns its input, then opens an IMPL library
        # on the same namespace to add a ZeroTensor kernel and checks a zero tensor input yields
        # a zero tensor output.
        with _scoped_library(self.test_ns, "DEF") as my_lib1:
            my_lib1.define("sum(Tensor self) -> Tensor")

            # Example 1
            @torch.library.impl(my_lib1, "sum", "CPU")
            def my_sum(*args, **kwargs):
                # [FUNCTION DOC] HELPER: CPU kernel returning a clone of the first argument.
                return args[0].clone()

            x = torch.tensor([1, 2])
            op = getattr(torch.ops, self.test_ns).sum
            self.assertEqual(op(x), x)

            with _scoped_library(self.test_ns, "IMPL") as my_lib2:
                # Example 2
                @torch.library.impl(my_lib2, op.default, "ZeroTensor")
                def my_sum_zt(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: ZeroTensor-key kernel: returns a fresh efficient
                    # zero tensor of the same shape when the input is a zero tensor, otherwise a
                    # clone of the input.
                    if args[0]._is_zerotensor():
                        return torch._efficientzerotensor(args[0].shape)
                    else:
                        return args[0].clone()

                y = torch._efficientzerotensor(3)
                self.assertTrue(op(y)._is_zerotensor())
                self.assertEqual(op(x), x)

    def test_create_new_library_fragment_no_existing(self):
        # [FUNCTION DOC] Defines an op in a FRAGMENT library when no library exists for the
        # namespace, registers a CPU kernel and checks the op is callable via torch.ops and
        # returns its input.
        with _scoped_library(self.test_ns, "FRAGMENT") as my_lib:
            my_lib.define("sum2(Tensor self) -> Tensor")

            @torch.library.impl(my_lib, "sum2", "CPU")
            def my_sum(*args, **kwargs):
                # [FUNCTION DOC] HELPER: CPU kernel returning its first argument.
                return args[0]

            x = torch.tensor([1, 2])
            self.assertEqual(getattr(torch.ops, self.test_ns).sum2(x), x)

    def test_create_new_library_fragment_with_existing(self):
        # [FUNCTION DOC] Defines ops in two nested FRAGMENT libraries on a namespace that
        # already has a DEF library, registers CPU kernels in each, and checks both ops are
        # callable and return their input.
        with _scoped_library(self.test_ns, "DEF") as my_lib1:
            # Create a fragment
            with _scoped_library(self.test_ns, "FRAGMENT") as my_lib2:
                my_lib2.define("sum4(Tensor self) -> Tensor")

                @torch.library.impl(my_lib2, "sum4", "CPU")
                def my_sum4(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: CPU kernel returning its first argument.
                    return args[0]

                x = torch.tensor([1, 2])
                self.assertEqual(getattr(torch.ops, self.test_ns).sum4(x), x)

                # Create another fragment
                with _scoped_library(self.test_ns, "FRAGMENT") as my_lib3:
                    my_lib3.define("sum3(Tensor self) -> Tensor")

                    @torch.library.impl(my_lib3, "sum3", "CPU")
                    def my_sum3(*args, **kwargs):
                        # [FUNCTION DOC] HELPER: CPU kernel returning its first argument.
                        return args[0]

                    x = torch.tensor([1, 2])
                    self.assertEqual(getattr(torch.ops, self.test_ns).sum3(x), x)

    @unittest.skipIf(IS_WINDOWS, "Skipped under Windows")
    def test_alias_analysis(self):
        # [FUNCTION DOC] Skipped on Windows. Runs a helper twice: with the default empty alias
        # analysis the TorchScript graph is expected to not contain the op (the helper raises
        # AssertionError, which the test expects); after a garbage collection, with CONSERVATIVE
        # alias analysis the op must appear in the scripted graph.
        def test_helper(alias_analysis=""):
            # [FUNCTION DOC] HELPER: Creates a library, defines an op with the given alias
            # analysis, scripts a function calling it, and raises AssertionError unless the op
            # name appears in the scripted graph text.
            my_lib1 = Library(self.test_ns, "DEF")  # noqa: SCOPED_LIBRARY

            called = [0]

            @torch.library.define(
                my_lib1, "_op() -> None", alias_analysis=alias_analysis
            )
            def _op(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Op implementation that increments a counter in the
                # enclosing list.
                called[0] += 1

            @torch.jit.script
                # [FUNCTION DOC] HELPER: Function compiled with TorchScript that calls the
                # registered test op; the test inspects its graph text, not its source.
            def _test():
                torch.ops._test_python_registration._op()

            if "_test_python_registration::_op" not in str(_test.graph):
                raise AssertionError("expected _test_python_registration::_op in graph")

        with self.assertRaises(AssertionError):
            test_helper("")  # alias_analysis="FROM_SCHEMA"

        # Run gc to make sure the previous Library is removed.  This is needed in dynamo-wrapped 3.14t
        gc.collect()
        test_helper("CONSERVATIVE")

    def test_error_for_unsupported_ns_or_kind(self) -> None:
        # [FUNCTION DOC] Checks Library construction errors: an unknown kind raises ValueError
        # about unsupported kind, and the reserved namespace prim with kinds DEF or FRAGMENT
        # raises ValueError about a reserved namespace.
        with self.assertRaisesRegex(ValueError, "Unsupported kind"):
            my_lib1 = Library("myns", "BLA")  # noqa: SCOPED_LIBRARY

        for kind in ("DEF", "FRAGMENT"):
            with self.assertRaisesRegex(ValueError, "reserved namespace"):
                my_lib1 = Library("prim", kind)  # noqa: SCOPED_LIBRARY

    def test_dispatcher_error_filenames(self) -> None:
        # [FUNCTION DOC] Creates one library and then a second library with the same namespace
        # to provoke a dispatcher error. Checks the error text omits the old placeholder path,
        # names this test file twice with line numbers, and that the source lines at those line
        # numbers (read back with linecache from this very file) contain the first and second
        # marker comments.
        # Test that dispatcher errors report correct Python filenames and line numbers
        # when defining duplicate libraries (which triggers the filename tracking)
        import linecache
        import re

        # Create first library
        # NOTE: Using Library directly instead of _scoped_library because this test
        # specifically verifies filename tracking in error messages, and _scoped_library
        # would report library.py locations instead of the actual test file locations
        lib1 = Library(self.test_ns, "DEF")  # FIRST_LIB_MARKER  # noqa: SCOPED_LIBRARY
        try:
            lib1.define("duplicate_op(Tensor x) -> Tensor")

            # Try to create another library with same namespace - this should trigger error
            with self.assertRaises(RuntimeError) as cm:
                lib2 = Library(  # SECOND_LIB_MARKER  # noqa: SCOPED_LIBRARY
                    self.test_ns, "DEF"
                )
        finally:
            lib1._destroy()

        error_msg = str(cm.exception)

        # The error should NOT contain /dev/null (the old placeholder)
        self.assertNotIn("/dev/null", error_msg)
        # The error should contain the test file name for both registrations
        self.assertIn("test_python_dispatch.py", error_msg)
        # Extract line numbers from the error message and verify they point to the right lines
        line_matches = re.findall(r"test_python_dispatch\.py:(\d+)", error_msg)
        self.assertEqual(
            len(line_matches), 2, "Should have exactly 2 line number references"
        )

        # Get the actual source lines and verify they contain our markers
        first_line_num, second_line_num = sorted([int(x) for x in line_matches])
        first_line = linecache.getline(__file__, first_line_num).strip()
        second_line = linecache.getline(__file__, second_line_num).strip()

        # Verify the lines contain our expected markers
        self.assertIn("FIRST_LIB_MARKER", first_line)
        self.assertIn("SECOND_LIB_MARKER", second_line)

    def test_returning_symint(self) -> None:
        # [FUNCTION DOC] Defines a custom op taking two SymInt and returning a SymInt,
        # implements it as the sum of squares, and calls it with the symbolic sizes of a
        # FakeTensor created under a ShapeEnv. Evaluates the returned symbolic expression and
        # checks it equals 13 (for shape 2 by 3).
        shape_env = ShapeEnv()
        fake_tensor_mode = FakeTensorMode(shape_env=shape_env)

        ft = fake_tensor_mode.from_tensor(torch.rand(2, 3))

        s0, s1 = ft.shape

        with _scoped_library(self.test_ns, "DEF") as tlib:
            tlib.define("sqsum(SymInt a, SymInt b) -> SymInt")

            @impl(tlib, "sqsum", "CompositeExplicitAutograd")
            def sqsum(a: SymInt, b: SymInt):
                # [FUNCTION DOC] HELPER: Composite kernel returning the sum of the squares of
                # its two symbolic ints.
                return a * a + b * b

            out = getattr(torch.ops, self.test_ns).sqsum.default(s0, s1)
            out_val = shape_env.evaluate_expr(out.node.expr)
        self.assertEqual(out_val, 13)

    def test_register_fallthrough(self):
        # [FUNCTION DOC] Registers a fallthrough kernel for aten mm on the CPU autocast key.
        # Under bfloat16 CPU autocast, mm must keep float32 while matmul becomes bfloat16; after
        # the library exits, mm autocasts to bfloat16 again.
        with _scoped_library("aten", "IMPL") as my_lib:
            my_lib.impl("mm", fallthrough_kernel, "AutocastCPU")

            a = torch.randn(2, 3, device="cpu", dtype=torch.float32)
            b = torch.randn(3, 2, device="cpu", dtype=torch.float32)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                # dtype for mm should be float32 since we registered a fallthrough
                self.assertEqual(torch.mm(a, b).dtype, torch.float32)
                # ops that don't have a fallthrough registered should not be affected
                self.assertEqual(torch.matmul(a, b).dtype, torch.bfloat16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # default behavior should have been restored
            self.assertEqual(torch.mm(a, b).dtype, torch.bfloat16)


instantiate_parametrized_tests(TestPythonRegistration)


class TestPythonDispatch(TestCase):
    # [CLASS DOC] Tests for tensor subclasses that implement __torch_dispatch__, mostly using
    # LoggingTensor and capture_logs to compare the sequence of dispatched aten ops against
    # inline expected strings. Also covers subclass creation errors, serialization, deepcopy,
    # modes, tracing with make_fx, FakeTensor, sizes/strides and more.
    def test_basic(self) -> None:
        # [FUNCTION DOC] Runs a multiplication of a requires-grad LoggingTensor and a backward
        # pass under log capture, checks the gradient value and saved tensor behavior under
        # no_grad (including in-place add), and compares the logged op sequence to the expected
        # text.
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input("x", x)
            y = x * x
            saved_x = y.grad_fn._saved_self
            grad_y = LoggingTensor(torch.tensor([1.0]))
            log_input("grad_y", grad_y)
            (g,) = torch.autograd.grad((y,), (x,), (grad_y,))

        self.assertEqual(g.elem, torch.tensor([6.0]))
        with torch.no_grad():
            self.assertEqual(saved_x, x)
            self.assertEqual(saved_x._version, x._version)
            x.add_(2)
            self.assertEqual(saved_x, x)
            # TODO: figure out why broken
            # self.assertEqual(saved_x._version, x._version)
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten.mul.Tensor($0, $0)
$2: f32[1] = input('grad_y')
$3: f32[1] = torch._ops.aten.mul.Tensor($2, $0)
$4: f32[1] = torch._ops.aten.mul.Tensor($2, $0)
$5: f32[1] = torch._ops.aten.add.Tensor($4, $3)""",
        )

    def test_out(self) -> None:
        # [FUNCTION DOC] Calls abs with an out argument on LoggingTensors, checks the out tensor
        # holds the result, and compares the log to the expected text showing the out keyword
        # being passed to the abs out overload.
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.zeros(1))
            log_input("x", x)
            log_input("y", y)
            torch.abs(x, out=y)

        self.assertEqual(y.elem, torch.ones(1))
        # TODO: arguably this shouldn't pass and we should complain
        # that out isn't a kwarg
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = input('y')
$2: f32[1] = torch._ops.aten.abs.out($0, out=$1)""",
        )

    def test_kwarg_only(self) -> None:
        # [FUNCTION DOC] Calls addmv with various beta/alpha keyword combinations on
        # LoggingTensors and checks the log omits beta and alpha when they equal defaults (even
        # if explicitly given) and shows them when non-default.
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.ones(1, 1))
            z = LoggingTensor(torch.ones(1))
            log_input("x", x)
            log_input("y", y)
            log_input("z", z)
            torch.addmv(x, y, z)
            torch.addmv(x, y, z, beta=1)
            torch.addmv(x, y, z, beta=2)
            torch.addmv(x, y, z, alpha=2)
            torch.addmv(x, y, z, beta=2, alpha=2)

        # The expectation is that beta/alpha don't show up when they're
        # defaulted.  This is even if the user explicitly specified it.
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1, 1] = input('y')
$2: f32[1] = input('z')
$3: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)
$4: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)
$5: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2)
$6: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, alpha=2)
$7: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2, alpha=2)""",
        )

    def test_kwarg_only_and_positional_default(self) -> None:
        # [FUNCTION DOC] Calls the private foobar op with positional and keyword arguments and
        # checks the log omits a defaulted middle positional argument when only a later
        # keyword-only argument is set, while an explicitly passed non-default-looking
        # positional value is kept.
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            log_input("x", x)
            torch.ops.aten._foobar(x)
            torch.ops.aten._foobar(x, False)
            torch.ops.aten._foobar(x, arg3=False)
            torch.ops.aten._foobar(x, False, arg3=False)

        # What we are testing here is that we omit arg2
        # if it is defaulted, even if a kwarg is set
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten._foobar.default($0)
$2: f32[1] = torch._ops.aten._foobar.default($0, False)
$3: f32[1] = torch._ops.aten._foobar.default($0, arg3=False)
$4: f32[1] = torch._ops.aten._foobar.default($0, False, arg3=False)""",
        )

    def test_produce_real_type(self) -> None:
        # [FUNCTION DOC] Exercises ops whose schemas have a plain dtype, an optional dtype and
        # an optional memory format (to copy, cumprod, contiguous after select) on a
        # LoggingTensor and compares the log, checking argument types are converted to proper
        # Python values such as torch dtypes and memory formats.
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(2, 2))
            log_input("x", x)
            x.to(dtype=torch.double)  # non-optional dtype
            torch.cumprod(x, 0, dtype=torch.double)  # optional dtype
            x[:, 1].contiguous(
                memory_format=torch.contiguous_format
            )  # optional memory format
            # There doesn't appear to be any layout signatures which are
            # triggerable using tensor subclasses (need to use a mode)

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[2, 2] = input('x')
$1: f64[2, 2] = torch._ops.aten._to_copy.default($0, dtype=torch.float64)
$2: f64[2, 2] = torch._ops.aten.cumprod.default($0, 0, dtype=torch.float64)
$3: f32[2] = torch._ops.aten.select.int($0, 1, 1)
$4: f32[2] = torch._ops.aten.clone.default($3, memory_format=torch.contiguous_format)""",
        )

    def test_optional_tensor_list(self) -> None:
        # [FUNCTION DOC] Registers a CPU kernel for a custom op taking a list of optional
        # tensors, calls it with None and a LoggingTensor, and checks the log shows the list
        # with None as a string and the tensor by its log id.
        def weird(xs):
            # [FUNCTION DOC] HELPER: Custom kernel that prints a marker and returns an empty
            # scalar tensor.
            print("woof")
            return torch.empty(())

        with _scoped_library("my_lib", "DEF") as my_lib:
            my_lib.define("weird(Tensor?[] self) -> Tensor")
            my_lib.impl("weird", weird, "CPU")
            with capture_logs() as logs:
                x = LoggingTensor(torch.ones(2, 2))
                log_input("x", x)
                torch.ops.my_lib.weird.default([None, x])

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[2, 2] = input('x')
$1: f32[] = torch._ops.my_lib.weird.default(['None', '$0'])""",
        )

    def test_list_ret(self) -> None:
        # [FUNCTION DOC] For both list and tuple return containers, defines a tensor subclass
        # whose dispatch hook returns the container of split results for the split op (raising
        # AssertionError for any other op), and checks the split of the subclass equals the
        # split of a plain tensor.
        # test all sequence types are permissible returns
        for list_type in (list, tuple):

            class A(torch.Tensor):
                # [CLASS DOC] HELPER: Tensor subclass created with the make-subclass helper; its
                # dispatch hook handles only the split op by returning a list or tuple (per the
                # enclosing loop variable) of the real split results computed with dispatch
                # disabled; other ops raise AssertionError.
                @staticmethod
                def __new__(cls, elem):
                    # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                    # requires_grad.
                    return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    # [FUNCTION DOC] HELPER: Dispatch hook for split as described by the class;
                    # raises AssertionError for unrecognized ops.
                    if func.overloadpacket == torch.ops.aten.split:
                        with no_dispatch():
                            return list_type(torch.split(*args))
                    else:
                        raise AssertionError(f"unrecognized func: {func}")

            self.assertEqual(
                torch.split(A(torch.tensor([0, 1])), 2),
                torch.split(torch.tensor([0, 1]), 2),
            )

    def test_invalid_ret(self) -> None:
        # [FUNCTION DOC] Defines a subclass whose dispatch hook returns a string and checks that
        # both neg and detach on it raise RuntimeError with message mentioning Unable to cast.
        # test invalid return gets reasonable error message
        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose dispatch hook always returns an invalid
            # result (a string), used to check error reporting.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                # requires_grad.
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that returns the string arf for any op.
                return "arf"

        # Wobbles depending on NDEBUG mode of pybind11
        self.assertRaisesRegex(
            RuntimeError,
            "Unable to cast",
            lambda: A(torch.zeros(1)).neg(),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Unable to cast",
            lambda: A(torch.zeros(1)).detach(),
        )

    def test_detach_appears_once_when_called_once(self) -> None:
        # [FUNCTION DOC] Calls detach once on a requires-grad LoggingTensor and checks the log
        # contains exactly the single detach shown in the expected text (a guard against
        # emitting even more detaches).
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input("x", x)
            x.detach()
        # FIXME: We actually want this to emit a single detach. However,
        # it currently emits two, for reasons unclear to us. Leaving
        # this test here to make sure we don't regress even further (it
        # would be bad if calling .detach() once emits 3+ detaches).
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten.detach.default($0)""",
        )

    def test_storage(self) -> None:
        # [FUNCTION DOC] Checks that taking the untyped storage of a LoggingTensor works and
        # that reading its data pointer raises RuntimeError.
        # For now, just make sure it doesn't crash.  Ideally, we should
        # return some virtual storage that is safe to work with
        x = LoggingTensor(torch.ones(1))
        storage = x.untyped_storage()
        self.assertRaises(RuntimeError, lambda: storage.data_ptr())

    def test_make_wrapper_subclass_noalloc(self) -> None:
        # [FUNCTION DOC] Creates a wrapper subclass with an enormous (about 8 TB) size and
        # checks it does not fail, since wrapper subclasses allocate no storage.
        # This is ludicrously big (8TB) and this should pass because wrapper
        # subclasses don't allocate
        torch.Tensor._make_wrapper_subclass(LoggingTensor, (1000000000000,))

    def test_version(self) -> None:
        # [FUNCTION DOC] Checks the version counter of a LoggingTensor: an in-place add on a
        # detached view bumps it, while an in-place add through the data attribute does not.
        x = LoggingTensor(torch.ones(1))
        prev_vc = x._version
        x.detach().add_(2)
        cur_vc = x._version
        self.assertNotEqual(prev_vc, cur_vc)
        x.data.add_(2)
        self.assertEqual(cur_vc, x._version)

    def test_subclass_priority(self) -> None:
        # [FUNCTION DOC] Defines two RuntimeError subclasses and a base subclass A plus derived
        # subclass B of it, each raising its own error in the dispatch hook. Checks that in
        # binary add the more derived subclass hook wins regardless of argument order.
        class ErrorA(RuntimeError):
            # [CLASS DOC] HELPER: Marker error raised by the base test subclass dispatch hook.
            pass

        class ErrorB(RuntimeError):
            # [CLASS DOC] HELPER: Marker error raised by the derived test subclass dispatch
            # hook.
            pass

        # The big tests for code coverage are test_precedence_semantics in
        # test_overrides.py; this is just to make sure it is wired up at all
        # correctly for __torch_dispatch__
        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Base tensor subclass whose dispatch hook always raises ErrorA.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                # requires_grad.
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that raises ErrorA.
                raise ErrorA

        class B(A):
            # [CLASS DOC] HELPER: Subclass of the base test tensor whose dispatch hook always
            # raises ErrorB.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                # requires_grad.
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that raises ErrorB.
                raise ErrorB

        self.assertRaises(
            ErrorA, lambda: torch.add(A(torch.empty(1)), A(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(A(torch.empty(1)), B(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(B(torch.empty(1)), A(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(B(torch.empty(1)), B(torch.empty(1)))
        )

    def test_format(self) -> None:
        # [FUNCTION DOC] Checks that str, repr and f-string formatting of a LoggingTensor agree
        # and equal the expected text.
        x = LoggingTensor(torch.ones(1))
        s1 = str(x)
        s2 = repr(x)
        s3 = f"{x}"
        self.assertExpectedInline(s1, """LoggingTensor(tensor([1.]))""")
        self.assertEqual(s1, s2)
        self.assertEqual(s1, s3)

    def test_custom_autograd(self) -> None:
        # [FUNCTION DOC] Runs a custom autograd Function (squaring, saving its input) on
        # LoggingTensors, checks the tensors seen in backward are LoggingTensors that stay equal
        # to the input, and compares the logged op sequence including the accumulation into the
        # preset grad.
        escape = [None]

        class Square(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd Function computing the square of its input;
            # forward saves the input, backward checks types and records the saved tensor.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Saves the input for backward and returns its square.
                y = x**2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Raises AssertionError unless the incoming gradient and
                # the saved input are LoggingTensors, stores the saved input in the enclosing
                # list, and returns the gradient times two times the input.
                if not isinstance(grad_output, LoggingTensor):
                    raise AssertionError(
                        f"expected LoggingTensor, got {type(grad_output)}"
                    )
                (x,) = ctx.saved_tensors
                if not isinstance(x, LoggingTensor):
                    raise AssertionError(f"expected LoggingTensor, got {type(x)}")
                escape[0] = x
                return grad_output * 2 * x

        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1), requires_grad=True)
            log_input("x", x)
            x.grad = LoggingTensor(torch.zeros(1))
            log_input("x.grad", x.grad)
            y = Square.apply(x)
            grad_output = LoggingTensor(torch.ones(1))
            log_input("grad_output", grad_output)
            y.backward(grad_output)

        with torch.no_grad():
            self.assertEqual(escape[0], x)
            self.assertEqual(escape[0]._version, x._version)
            # TODO: figure out why x.requires_grad = False doesn't
            # trigger an error for LoggingTensor
            x.add_(2)
            self.assertEqual(escape[0], x)
            # TODO: figure out why this is broken
            # self.assertEqual(escape[0]._version, x._version)

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = input('x.grad')
$2: f32[1] = torch._ops.aten.pow.Tensor_Scalar($0, 2)
$3: f32[1] = input('grad_output')
$4: f32[1] = torch._ops.aten.mul.Tensor($3, 2)
$5: f32[1] = torch._ops.aten.mul.Tensor($4, $0)
$6: f32[1] = torch._ops.aten.add_.Tensor($1, $5)""",
        )

    def test_subclass_creation(self):
        # [FUNCTION DOC] Checks that creating a plain Tensor subclass from or around a tensor
        # already tied to a LoggingTensor Python object raises RuntimeError (via make-subclass,
        # as-subclass and direct construction), and that making a wrapper subclass of a class
        # lacking a dispatch hook raises TypeError.
        # Make sure these statements runs without error
        # In particular checking that when internal detach returns
        # subclasses, these are cleanly overwritten.
        class Foo(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass with no dispatch hook.
            pass

        err_msg = "subclass Foo but.*already associated to a python object of type LoggingTensor"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            a = torch.Tensor._make_subclass(Foo, LoggingTensor(torch.rand(2)))
        with self.assertRaisesRegex(RuntimeError, err_msg):
            b = LoggingTensor(torch.rand(2)).as_subclass(Foo)
        with self.assertRaisesRegex(RuntimeError, err_msg):
            Foo(LoggingTensor(torch.rand(2)))

        with self.assertRaisesRegex(TypeError, "Foo must define __torch_dispatch__"):
            torch.Tensor._make_wrapper_subclass(Foo, (2, 2))

    def test_new_ones(self) -> None:
        # [FUNCTION DOC] Checks that new_ones on a subclass whose dispatch hook returns a
        # subclass instance gives back that subclass type.
        class MyTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose dispatch hook returns a new instance of
            # itself for any op.
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook returning a fresh instance of the class
                # regardless of the op.
                return MyTensor(3)

        self.assertEqual(type(MyTensor(2).new_ones(3)), MyTensor)

    def test_like(self) -> None:
        # [FUNCTION DOC] Checks that the like-style factory functions (empty, ones, rand, randn,
        # zeros, full, randint) applied to a subclass instance return the subclass type.
        class MyTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose dispatch hook returns a new instance of
            # itself for any op.
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook returning a fresh instance of the class
                # regardless of the op.
                return MyTensor(3)

        for f in ["empty", "ones", "rand", "randn", "zeros"]:
            f_name = f + "_like"
            self.assertEqual(type(getattr(torch, f_name)(MyTensor(2))), MyTensor)

        self.assertEqual(type(torch.full_like(MyTensor(2), 1.0)), MyTensor)
        self.assertEqual(type(torch.randint_like(MyTensor(2), high=3)), MyTensor)

    def test_traceable_wrapper_subclass_protocol_runtime_check(self) -> None:
        # [FUNCTION DOC] Checks the runtime predicates for the traceable wrapper subclass
        # protocol on instance and type levels (false for plain tensors, true for both static
        # and classmethod unflatten wrappers) and that flatten followed by unflatten rebuilds
        # the same wrapper type with an equal inner tensor. The checks run inside a
        # Dynamo-disabled helper.
        @torch._dynamo.disable
        def run_checks() -> None:
            # [FUNCTION DOC] HELPER: Dynamo-disabled body performing all the protocol checks
            # described by the test.
            base = torch.randn(2, 2)
            static = _StaticUnflattenWrapper(base)
            classmethod_wrapper = _ClassmethodUnflattenWrapper(base)

            self.assertFalse(is_traceable_wrapper_subclass(base))
            self.assertTrue(is_traceable_wrapper_subclass(static))
            self.assertTrue(is_traceable_wrapper_subclass(classmethod_wrapper))
            self.assertTrue(is_traceable_wrapper_subclass_type(type(static)))
            self.assertTrue(
                is_traceable_wrapper_subclass_type(type(classmethod_wrapper))
            )
            self.assertFalse(is_traceable_wrapper_subclass_type(torch.Tensor))

            static_attrs, static_meta = static.__tensor_flatten__()
            static_rebuilt = type(static).__tensor_unflatten__(
                {attr: getattr(static, attr) for attr in static_attrs},
                static_meta,
                static.size(),
                static.stride(),
            )
            self.assertIs(type(static_rebuilt), _StaticUnflattenWrapper)
            self.assertEqual(static_rebuilt.elem, static.elem)

            classmethod_attrs, classmethod_meta = (
                classmethod_wrapper.__tensor_flatten__()
            )
            classmethod_rebuilt = type(classmethod_wrapper).__tensor_unflatten__(
                {
                    attr: getattr(classmethod_wrapper, attr)
                    for attr in classmethod_attrs
                },
                classmethod_meta,
                classmethod_wrapper.size(),
                classmethod_wrapper.stride(),
            )
            self.assertIs(type(classmethod_rebuilt), _ClassmethodUnflattenWrapper)
            self.assertEqual(classmethod_rebuilt.elem, classmethod_wrapper.elem)

        run_checks()

    def test_make_fx_with_subclass(self) -> None:
        # [FUNCTION DOC] Traces a function over a TwoTensor and a plain tensor with make_fx in
        # fake mode, flattening the subclass output manually, and compares the generated graph
        # code (two muls and an add on the inner tensors) to the expected text.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the product of x and y and the sum y plus y.
            # Returns (TwoTensor, Tensor)
            return x * y, y + y

        x_a = torch.zeros(4)
        x_b = torch.zeros(4)
        y = torch.ones(4)

        # make_fx() is not responsible for unwrapping tensor subclass inputs,
        # so we do it manually here.
        # Why? In general, make_fx(f)(*args) promises that the graph returned has the same calling
        # convention as f(*args). Unwrapping tensor subclass inputs can potentially change
        # the number of input args to the graph, breaking that assumption
        def f_to_trace(x_a, x_b, y):
            # [FUNCTION DOC] HELPER: Wraps two plain tensors in a TwoTensor, applies f, and
            # returns the inner tensors of the first output followed by the second output, so
            # the traced graph has a plain-tensor calling convention.
            x = TwoTensor(x_a, x_b)
            out1, out2 = f(x, y)
            out1_unwrapped_attrs, _ = out1.__tensor_flatten__()
            return (*[getattr(out1, attr) for attr in out1_unwrapped_attrs], out2)

        fx_g = make_fx(f_to_trace, tracing_mode="fake")(x_a, x_b, y)
        self.assertExpectedInline(
            fx_g.code,
            """\



def forward(self, x_a_1, x_b_1, y_1):
    mul = torch.ops.aten.mul.Tensor(x_a_1, y_1);  x_a_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(x_b_1, y_1);  x_b_1 = None
    add = torch.ops.aten.add.Tensor(y_1, y_1);  y_1 = None
    return (mul, mul_1, add)
    """,
        )

    # See https://github.com/pytorch/pytorch/issues/117794
    def test_return_and_correct_aliasing_gives_correct_stride(self):
        # [FUNCTION DOC] Regression test for issue 117794: slicing a column of a TwoTensor must
        # give the same stride as slicing a dense tensor of the same shape.
        t = TwoTensor(torch.randn(2, 2), torch.randn(2, 2))
        x = torch.randn(2, 2)
        # slicing should result in the same stride for TwoTensor as a dense tensor would give
        self.assertEqual(t[:, 0].stride(), x[:, 0].stride())

    def test_make_wrapper_subclass_propagates_metadata(self) -> None:
        # [FUNCTION DOC] Builds a wrapper subclass over a non-contiguous, offset diagonal view
        # and checks the wrapper reports the same size, stride and storage offset.
        class WrapperTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Local wrapper tensor subclass holding elem in a slot and
            # copying its size, strides, dtype, layout, device, requires_grad and storage
            # offset; its dispatch hook raises RuntimeError.
            elem: torch.Tensor

            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the wrapper via the make-wrapper-subclass hook
                # with metadata copied from elem and stores elem.
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that always raises RuntimeError with
                # message NYI.
                raise RuntimeError("NYI")

        # non-contiguous strides, non-zero storage offset
        x = torch.randn(4, 6).t().diagonal(offset=2)
        y = WrapperTensor(x)
        self.assertEqual(y.size(), x.size())
        self.assertEqual(y.stride(), x.stride())
        self.assertEqual(y.storage_offset(), x.storage_offset())

    def test_wrapper_subclass_serializes(self) -> None:
        # [FUNCTION DOC] Saves a LoggingTensor of int64 values to a temporary file, reloads it
        # with the class allowed in safe globals, and checks the loaded object has the same
        # type, equal values and equal inner tensor but is a distinct object.
        with tempfile.TemporaryFile() as f:
            # purposefully use int64 to test non-default dtype
            x = LoggingTensor(torch.randperm(3))
            torch.save(x, f)
            f.seek(0)
            with torch.serialization.safe_globals([LoggingTensor]):
                x_loaded = torch.load(f)
            self.assertTrue(type(x_loaded) is type(x))
            self.assertEqual(x, x_loaded)
            self.assertEqual(x.elem, x_loaded.elem)
            self.assertFalse(x is x_loaded)

    def test_deepcopy_wrapper_subclass(self) -> None:
        # [FUNCTION DOC] Deep-copies a LoggingTensor of int64 values and checks the copy has the
        # same type, equal values and inner tensor and is a distinct object.
        # purposefully use int64 to test non-default dtype
        x = LoggingTensor(torch.randperm(3))
        x_copy = deepcopy(x)
        self.assertTrue(type(x_copy) is type(x))
        self.assertEqual(x, x_copy)
        self.assertEqual(x.elem, x_copy.elem)
        self.assertFalse(x is x_copy)

    def test_deepcopy_wrapper_subclass_with_clone_returning_different_type(
        self,
    ) -> None:
        # [FUNCTION DOC] Deep-copies a wrapper subclass whose clone returns a plain tensor and
        # checks RuntimeError is raised with a message saying cloning must return another
        # instance of the same subclass.
        class MyWrapperTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass like the others, but its dispatch hook
            # returns a plain clone of the inner tensor for the clone op and raises RuntimeError
            # otherwise.
            elem: torch.Tensor

            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the wrapper via the make-wrapper-subclass hook
                # with metadata copied from elem and stores elem.
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook returning the clone of the inner tensor
                # when the op is clone, else raising RuntimeError with message NYI.
                if func.overloadpacket.__name__ == "clone":
                    # Return a plain tensor from clone().
                    return args[0].elem.clone()
                raise RuntimeError("NYI")

            # NB: The default Tensor.__torch_function__ implementation called for deepcopy
            # disables __torch_function__ by the time we get to clone(), so there is no need to
            # explicitly disable __torch_function__ for this subclass.

        x = MyWrapperTensor(torch.randn(3))
        with self.assertRaisesRegex(
            RuntimeError,
            "for which cloning returns another instance of the same subclass",
        ):
            x_copy = deepcopy(x)

    def test_deepcopy_non_wrapper_subclass(self) -> None:
        # [FUNCTION DOC] Checks deepcopy errors: for a subclass using the default new_empty and
        # one whose new_empty returns a plain tensor, deepcopy raises RuntimeError saying the
        # function must return another instance of the same subclass; a correct new_empty
        # subclass is then expected to deep-copy successfully.
        # Ensure correct error is thrown for common error cases.
        class SubTensorError1(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass relying on the default new_empty, which
            # returns a plain tensor.
            # Default implementation of new_empty() returns a plain tensor.
            pass

        class SubTensorError2(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose new_empty wrongly returns a plain
            # tensor.
            # new_empty() incorrectly returns a different type (i.e. a plain tensor).
            def new_empty(self, shape):
                # [FUNCTION DOC] HELPER: Returns a plain tensor built from shape, which is the
                # deliberate bug.
                return torch.Tensor(shape)

        for error_cls in [SubTensorError1, SubTensorError2]:
            x = error_cls(3)
            with self.assertRaisesRegex(
                RuntimeError,
                "for which that function returns another instance of the same subclass",
            ):
                x_copy = deepcopy(x)

        # Ensure a correctly implemented new_empty() causes deepcopy() to work.
        class SubTensorSuccess(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass with a new_empty meant to return an instance
            # of its own class.
            def new_empty(self, shape):
                # [FUNCTION DOC] HELPER: Returns a new instance of the same class built from
                # shape.
                return type(self)(shape)

        x = SubTensorSuccess(3)
        x_copy = deepcopy(x)
        self.assertIs(type(x_copy), type(x))

    def test_wrapper_subclass_extra_dispatch_keys(self) -> None:
        # [FUNCTION DOC] Creates a wrapper subclass through the positional overload of the
        # private make-wrapper-subclass hook, passing extra dispatch keys containing the
        # NestedTensor key. Checks the instance has the NestedTensor key but not the autograd
        # NestedTensor key.
        class ExtraKeysTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass created with an explicit extra
            # dispatch key set (NestedTensor); its dispatch hook does nothing.
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Builds the wrapper using the positional overload (size,
                # stride, offset, memory format, dtype, layout, device, flags and the extra key
                # set) and returns it; elem is not stored.
                # NB: only the non-kwarg overload of _make_wrapper_subclass supports
                #     extra dispatch keys. We probably want to unify the two APIs
                #     in the future.
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    elem.stride(),
                    elem.storage_offset(),
                    torch.contiguous_format,
                    elem.dtype,
                    elem.layout,
                    elem.device,
                    False,
                    False,
                    None,
                    False,
                    False,
                    DispatchKeySet(DispatchKey.NestedTensor),
                )
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that does nothing and returns None.
                pass

        x = ExtraKeysTensor(torch.randn(3))
        self.assertTrue(torch._C._dispatch_keys(x).has(DispatchKey.NestedTensor))
        self.assertFalse(
            torch._C._dispatch_keys(x).has(DispatchKey.AutogradNestedTensor)
        )

    def test_wrapper_subclass_multiprocessing_preserves_dtype(self):
        # [FUNCTION DOC] Builds a TwoTensor of int64 tensors, passes it through a DataLoader
        # with two worker processes and an identity collate function, and checks the batch
        # element received in the parent keeps the original dtype (not the default float dtype).
        # a and b have dtype of int64, which is purposefully different from the default
        # assumed by _make_wrapper_subclass().
        a = torch.randperm(5)
        b = torch.randperm(5)
        data = TwoTensor(a, b)
        expected_dtype = data.dtype

        loader = torch.utils.data.DataLoader(
            [data, data],
            batch_size=2,
            num_workers=2,
            collate_fn=_identity,
        )
        for batch in loader:
            self.assertEqual(batch[0].dtype, expected_dtype)

    def test_index_put_where_only_index_is_subclass(self) -> None:
        # [FUNCTION DOC] Calls in-place index_put on a plain tensor with a tuple of indices
        # whose only element is a wrapper subclass, and checks the subclass dispatch hook saw
        # exactly one call, to the default overload of index_put_.
        called_funcs = []

        class MyTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass that records every dispatched op in an
            # enclosing list and returns a new wrapper around a scalar tensor.
            elem: torch.Tensor
            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the wrapper through the keyword
                # make-wrapper-subclass hook with dtype, layout, device and requires_grad from
                # elem, and stores elem.
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Appends the dispatched op to the enclosing list and
                # returns a wrapper around a scalar tensor.
                called_funcs.append(func)
                return MyTensor(torch.tensor(3))

        x = torch.randn(3, 3)
        idxs = (MyTensor(torch.tensor(0)),)
        v = torch.randn(1)
        res = x.index_put_(idxs, v)
        self.assertEqual(called_funcs, [torch.ops.aten.index_put_.default])

    def test_torch_dispatch_mode_basic(self) -> None:
        # [FUNCTION DOC] Under a LoggingTensorMode with log capture, creates an empty tensor and
        # compares the log to the expected single empty-op line.
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                torch.empty([])
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)""",
        )

    def test_torch_dispatch_mode_unrelated_tensors(self) -> None:
        # [FUNCTION DOC] Adds two plain tensors under a LoggingTensorMode with log capture and
        # checks the log has the add line, showing the mode logs ops on tensors it did not
        # create.
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                x + y
        self.assertExpectedInline(
            "\n".join(logs), """$2: f32[] = torch._ops.aten.add.Tensor($0, $1)"""
        )

    def test_nested_push_logging_tensor_mode(self):
        # [FUNCTION DOC] Nests two LoggingTensorModes and runs an empty and an add; the expected
        # log contains each op twice (once per mode layer).
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                with LoggingTensorMode():
                    torch.empty([])
                    x + y

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

    def test_capture_logs_with_torch_dispatch_mode(self):
        # [FUNCTION DOC] Checks the helper that captures logs with a logging tensor mode: a
        # single capture logs each op once, and two nested captures log each op twice with both
        # log lists equal.
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs_with_logging_tensor_mode() as logs:
            torch.empty([])
            x + y
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

        x = torch.randn([])
        y = torch.randn([])

        with capture_logs_with_logging_tensor_mode() as logs1:
            with capture_logs_with_logging_tensor_mode() as logs2:
                torch.empty([])
                x + y

        self.assertExpectedInline(
            "\n".join(logs2),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

        self.assertEqual(logs1, logs2)

    def test_torch_dispatch_mode_subclass_priority(self) -> None:
        # [FUNCTION DOC] Checks precedence between subclasses and modes: with subclass hooks
        # that enter a mode and raise, a derived subclass wins over its base, and an active mode
        # takes precedence over argument subclasses. Expected errors are asserted for each
        # combination of A, B, AMode and BMode.
        class ErrorA(RuntimeError):
            # [CLASS DOC] HELPER: Marker RuntimeError raised by the A-side hooks.
            pass

        class ErrorB(RuntimeError):
            # [CLASS DOC] HELPER: Marker RuntimeError raised by the B-side hooks.
            pass

        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose dispatch hook enters AMode and raises
            # ErrorA.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                # requires_grad.
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Enters AMode and raises ErrorA (AMode itself raises
                # ErrorA first).
                with AMode():
                    raise ErrorA

        class B(A):
            # [CLASS DOC] HELPER: Derived subclass of A whose dispatch hook enters BMode and
            # re-runs the op, so BMode raises ErrorB.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                # requires_grad.
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Enters BMode and calls the op with the given args,
                # which triggers ErrorB from the mode.
                with BMode():
                    func(*args, **kwargs)

        class AMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that raises ErrorA for every op.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Raises ErrorA.
                raise ErrorA

        class BMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that raises ErrorB for every op.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Raises ErrorB.
                raise ErrorB

        a = A(torch.empty(1))
        b = B(torch.empty(1))
        with self.assertRaises(ErrorA):
            a + a
        with self.assertRaises(ErrorB):
            a + b

        # B has precedence over A due to the subclass relationship yet
        # modes take precedence over arguments
        with self.assertRaises(ErrorA):
            with AMode():
                b + b
        with self.assertRaises(ErrorB):
            with BMode():
                a + a
        with self.assertRaises(ErrorB):
            with BMode():
                a + b

    def test_mode_with_make_subclass(self):
        # [FUNCTION DOC] Creates a subclass instance with the make-subclass helper while a
        # pass-through dispatch mode is active, and checks the result is an instance of that
        # subclass.
        class SubTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Plain Tensor subclass built via the make-subclass helper;
            # defines no dispatch hook.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                # requires_grad.
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        class BasicMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Pass-through dispatch mode that simply runs the op.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the op with the given args and kwargs and returns
                # its result.
                return func(*args, **kwargs)

        x = torch.randn(3)
        with BasicMode():
            y = SubTensor(x)
        self.assertIsInstance(y, SubTensor)

    def test_torch_dispatch_mode_respects_no_dispatch(self) -> None:
        # [FUNCTION DOC] Checks that ops run inside the no-dispatch context while a
        # LoggingTensorMode is active are not logged: the log with a nested no_dispatch block
        # equals the log without it.
        with capture_logs(is_mode=True) as logs1:
            with LoggingTensorMode():
                torch.ones([2, 3])
                with no_dispatch():
                    torch.ones([2, 3])
        with capture_logs(is_mode=True) as logs2:
            with LoggingTensorMode():
                torch.ones([2, 3])
        self.assertEqual(logs1, logs2)

    def test_shallow_copy_and_detach(self) -> None:
        # [FUNCTION DOC] Runs a forward and backward pass under a mode that asserts every tensor
        # argument was previously seen as an output of the mode (so detach and shallow-copy
        # results flow through the mode) and records each result.
        seen = set()
        test_case = self

        class TestMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that asserts all tensors in its args were
            # already seen, runs the op, and records the result tensors in the enclosing set.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Checks every tensor in args and kwargs is in the seen
                # set, runs the op (with empty kwargs if None), adds result tensors to the set
                # and returns the result.
                tree_map_only(
                    torch.Tensor, lambda t: test_case.assertIn(t, seen), (args, kwargs)
                )
                if kwargs is None:
                    kwargs = {}
                r = func(*args, **kwargs)
                tree_map_only(torch.Tensor, lambda t: seen.add(t), r)
                return r

        with TestMode():
            x = torch.randn(3, requires_grad=True)
            loss = (x * x).sum()
            loss.backward()

    def test_exception_handling(self):
        # [FUNCTION DOC] Checks that a mode survives an exception raised by its own dispatch
        # hook: after a caught RuntimeError from randn, a later zeros call inside the same mode
        # still returns the subclass.
        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Plain Tensor subclass built via the make-subclass helper.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                # requires_grad.
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        class AMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode raising RuntimeError for randn and returning a
            # subclass instance of zeros for every other op.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Raises RuntimeError when the op is randn, otherwise
                # returns a subclass instance holding a scalar zero.
                if func.__name__ == "randn.default":
                    raise RuntimeError
                return A(torch.zeros(()))

        with AMode():
            try:
                torch.randn(())
            except RuntimeError:
                pass
            self.assertTrue(isinstance(torch.zeros(()), A))

    def test_with_mode_created_separately(self):
        # [FUNCTION DOC] Creates a mode object first and enters it later with a with statement;
        # checks its dispatch hook error propagates out of an empty call.
        class ErrorA(RuntimeError):
            # [CLASS DOC] HELPER: Marker RuntimeError raised by the mode.
            pass

        class A(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that always raises ErrorA.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Raises ErrorA.
                raise ErrorA

        x = A()
        with self.assertRaises(ErrorA):
            with x:
                torch.empty([])

    def test_with_nested_modes(self):
        # [FUNCTION DOC] Nests two modes that raise an error carrying their own message and
        # checks the error seen is from the innermost mode (layer2).
        class ErrorA(RuntimeError):
            # [CLASS DOC] HELPER: RuntimeError subclass constructed with a message.
            def __init__(self, msg):
                # [FUNCTION DOC] HELPER: Passes msg to the base exception constructor.
                super().__init__(msg)

        class A(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode storing a message and raising ErrorA with it for
            # any op.
            def __init__(self, msg):
                # [FUNCTION DOC] HELPER: Stores msg as an attribute. Does not call the base
                # constructor.
                self.msg = msg

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Raises ErrorA carrying the stored message.
                raise ErrorA(self.msg)

        with self.assertRaisesRegex(ErrorA, "layer2"):
            with A("layer1"):
                with A("layer2"):
                    torch.empty([])

    def test_make_subclass_with_modes(self):
        # [FUNCTION DOC] Checks that a dispatch mode can wrap results into a tensor subclass
        # that remembers the mode, and that a subclass built with the make-subclass helper
        # cannot be nested under a second mode. Ends with an assertRaisesRegex expression that
        # is not entered as a context manager.
        # # KNOWN ISSUE: the final guard calls assertRaisesRegex without a callable or with
        # statement, so it returns a context manager object that is always true and the raise
        # branch never runs; that last check tests nothing.
        class ModeTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass created via the make-subclass helper that
            # remembers its wrapped tensor and the mode that created it; its own dispatch hook
            # must never run.
            def __new__(cls, elem, mode):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem and sets
                # attributes for elem and mode.
                r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
                r.elem = elem
                r.mode = mode
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Raises NotImplementedError; it should never be reached.
                raise NotImplementedError("Shouldn't be here")

        class Mode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that unwraps ModeTensor arguments, runs the op,
            # and wraps tensor results into ModeTensor tied to the mode.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Unwraps positional ModeTensor arguments, runs the op
                # with the given kwargs and wraps the result.
                def unwrap(e):
                    # [FUNCTION DOC] HELPER: Returns the inner tensor for a ModeTensor,
                    # otherwise the value as is.
                    if isinstance(e, ModeTensor):
                        return e.elem
                    else:
                        return e

                def wrap(t):
                    # [FUNCTION DOC] HELPER: Wraps a tensor into a ModeTensor tied to the mode;
                    # other values are returned unchanged.
                    if isinstance(t, torch.Tensor):
                        return ModeTensor(t, self)
                    else:
                        return t

                return wrap(func(*tuple(unwrap(a) for a in args), **kwargs))

        class BasicMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Pass-through dispatch mode that simply runs the op.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the op with the given args and kwargs and returns
                # its result.
                return func(*args, **kwargs)

        x = torch.tensor(4.0)
        with Mode():
            y = x + x
            z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)

        with Mode():
            with BasicMode():  # we can't nest two modes that call make_subclass because it only accepts vanilla tensors
                y = x + x
                z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)

        if not self.assertRaisesRegex(
            RuntimeError,
            "subclass Mode but.* associated to a python object of type Mode",
        ):
            raise AssertionError("expected RuntimeError")

    def test_notimplemented_mode(self):
        # [FUNCTION DOC] Checks a mode returning NotImplemented for subclass types: the polite
        # mode sees two dispatch attempts (one declined), the wrapper subclass sees one, and the
        # post-decline count is one; nesting two polite modes must not raise.
        sub_count = 0

        class PoliteMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode counting attempts (pre_count) and accepted calls
            # (post_count). It returns NotImplemented if any participating type is not exactly
            # Tensor, else runs the op.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes both counters to zero.
                self.pre_count = 0
                self.post_count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Increments pre_count; returns NotImplemented if any
                # type in types is not plain Tensor; otherwise increments post_count and runs
                # the op.
                self.pre_count += 1
                if any(t is not torch.Tensor for t in types):
                    return NotImplemented
                self.post_count += 1
                return func(*args, **kwargs)

        class SubTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass over elem that counts its dispatch calls in
            # the enclosing variable and runs ops on unwrapped args.
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates a wrapper with the same shape as elem and
                # stores elem.
                r = torch.Tensor._make_wrapper_subclass(cls, elem.shape)
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Increments the enclosing counter, unwraps subclass
                # instances in args and kwargs with a tree map and runs the op.
                nonlocal sub_count
                sub_count += 1

                def unwrap(t):
                    # [FUNCTION DOC] HELPER: Returns the inner tensor of a SubTensor, otherwise
                    # the value as is.
                    if isinstance(t, SubTensor):
                        return t.elem
                    else:
                        return t

                return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        a = SubTensor(torch.randn(2))
        with PoliteMode() as mode:
            a.abs()

        self.assertEqual(mode.pre_count, 2)
        self.assertEqual(mode.post_count, 1)
        self.assertEqual(sub_count, 1)

        # make sure this doesn't error
        with PoliteMode():
            with PoliteMode():
                a.abs()

    def test_nesting_same_mode(self):
        # [FUNCTION DOC] Enters a logging mode and re-enters the same instance; checks the same
        # mode instance may be pushed again and that the log shows the empty op twice.
        # If the pushed mode is the same instance as the current mode, we allow pushing an already active mode.

        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode() as reenabled:
                with reenabled:
                    torch.empty([])
            self.assertExpectedInline(
                "\n".join(logs),
                """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)""",
            )

    def test_error_using_class_method_on_mode(self):
        # [FUNCTION DOC] Checks that defining the dispatch hook of a mode as a classmethod
        # raises RuntimeError saying classmethod is not supported.
        class A(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that wrongly defines its hook as a classmethod.
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Classmethod hook that calls the op with the args and
                # kwargs passed as two single arguments; it is never successfully used because
                # mode dispatch rejects classmethods.
                return func(args, kwargs)

        x = torch.tensor(5.0)
        with self.assertRaisesRegex(
            RuntimeError, "classmethod is not supported, please make it a plain method"
        ):
            with A():
                x + x

    def test_get_cur_mode(self):
        # [FUNCTION DOC] Checks the current-dispatch-mode helper returns None outside modes and
        # returns the innermost active mode, including a re-entered mode and a nested one.
        class A(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode with a do-nothing hook.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Does nothing and returns None.
                pass

        self.assertEqual(_get_current_dispatch_mode(), None)

        with A() as mode1:
            self.assertEqual(_get_current_dispatch_mode(), mode1)

        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_dispatch_mode(), mode2)

    def test_get_mode_stack(self):
        # [FUNCTION DOC] Checks the dispatch mode stack helper returns an empty list with no
        # modes and lists active modes from outermost to innermost, including a re-entered mode.
        class A(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode with a do-nothing hook.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Does nothing and returns None.
                pass

        self.assertEqual(_get_current_dispatch_mode_stack(), [])

        with A() as mode1:
            self.assertEqual(_get_current_dispatch_mode_stack(), [mode1])

        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_dispatch_mode_stack(), [mode1, mode2])

    def test_all_same_mode(self):
        # [FUNCTION DOC] Checks all_same_mode is true for repeated references to one mode, false
        # when a None is present, and false for two distinct mode instances.
        x = LoggingTensorMode()
        y = LoggingTensorMode()
        self.assertTrue(all_same_mode([x, x, x]))
        self.assertFalse(all_same_mode([x, None]))
        self.assertFalse(all_same_mode([x, y]))

    def test_mode_detection(self):
        # [FUNCTION DOC] Nests infra and non-infra dispatch modes and checks
        # is_in_torch_dispatch_mode with and without infra modes included at each nesting level
        # and after leaving them.
        class InfraMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that declares itself an infra mode through its
            # classmethod.
            @classmethod
            def is_infra_mode(cls):
                # [FUNCTION DOC] HELPER: Returns True to mark the class as an infra mode.
                return True

        class NonInfraMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Ordinary dispatch mode with inherited, non-infra behavior.
            pass

        with InfraMode():
            self.assertTrue(is_in_torch_dispatch_mode())
            self.assertFalse(is_in_torch_dispatch_mode(include_infra_modes=False))
            with NonInfraMode():
                self.assertTrue(is_in_torch_dispatch_mode())
                self.assertTrue(is_in_torch_dispatch_mode(include_infra_modes=False))
                with InfraMode():
                    self.assertTrue(is_in_torch_dispatch_mode())
                    self.assertTrue(
                        is_in_torch_dispatch_mode(include_infra_modes=False)
                    )

                self.assertTrue(is_in_torch_dispatch_mode())
                self.assertTrue(is_in_torch_dispatch_mode(include_infra_modes=False))
            self.assertTrue(is_in_torch_dispatch_mode())
            self.assertFalse(is_in_torch_dispatch_mode(include_infra_modes=False))

        self.assertFalse(is_in_torch_dispatch_mode())
        self.assertFalse(is_in_torch_dispatch_mode(include_infra_modes=False))

    def test_tolist_numpy_with_torch_dispatch_mode(self) -> None:
        # [FUNCTION DOC] Checks tolist and numpy on a LoggingTensor raise RuntimeError saying
        # the operation is unsupported for tensor subclasses, and that comparing the subclass
        # with None through assertEqual raises AssertionError.
        x = LoggingTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(
            RuntimeError, "is not supported for tensor subclasses."
        ):
            x.tolist()
        with self.assertRaisesRegex(
            RuntimeError, "is not supported for tensor subclasses."
        ):
            x.numpy()
        with self.assertRaises(AssertionError):
            self.assertEqual(x, None)

    # See https://github.com/pytorch/pytorch/issues/136064
    def test_view_returns_alias_under_torch_dispatch(self):
        # [FUNCTION DOC] Regression test for issue 136064: under a mode, the view-by-dtype op
        # must return a new tensor object distinct from its input, which the mode asserts after
        # running the op.
        class MyMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode holding a reference to the test case. It runs
            # each op and, for the dtype view overload, asserts the output is a different object
            # than the input.
            def __init__(self, testcase):
                # [FUNCTION DOC] HELPER: Stores the test case.
                self.testcase = testcase

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the op, asserts for the view-dtype overload that
                # the output is not the first argument, and returns the output.
                out = func(*args, **kwargs)
                if func == torch.ops.aten.view.dtype:
                    # view should return a fresh TensorImpl
                    self.testcase.assertTrue(out is not args[0])
                return out

        with MyMode(self):
            x = torch.ones(4, dtype=torch.float32)
            out = x.view(torch.float32)

    def test_record_stream(self) -> None:
        # [FUNCTION DOC] Calls record_stream with a Stream object on a tensor under a mode that
        # checks the op name, the argument types and the stream id, device index and device type
        # with 1, 2 and 3.
        class TestMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that asserts the dispatched op is record_stream
            # and that its stream argument carries the expected id, device index and device
            # type.
            def __init__(self, testcase):
                # [FUNCTION DOC] HELPER: Stores the test case.
                self.testcase = testcase

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Performs the assertions about the op name, tensor and
                # stream arguments; returns None.
                self.testcase.assertEqual(func.name(), "aten::record_stream")
                self.testcase.assertIsInstance(args[0], torch.Tensor)
                self.testcase.assertIsInstance(args[1], torch.Stream)
                self.testcase.assertEqual(args[1].stream_id, 1)
                self.testcase.assertEqual(args[1].device_index, 2)
                self.testcase.assertEqual(args[1].device_type, 3)

        t = torch.tensor(5.0)
        s = torch.Stream(stream_id=1, device_index=2, device_type=3)
        with TestMode(self):
            t.record_stream(s)

    def test_return_stream(self) -> None:
        # [FUNCTION DOC] Defines a custom op returning a Stream, with a CPU kernel returning
        # stream values 0, 1 and 2. Checks the normal call returns that stream, and that under a
        # dispatch mode returning a different stream with 1, 2 and 3 the mode result is used.
        with _scoped_library("test_return_stream", "DEF") as l_def:
            l_def.define("return_stream(Tensor self) -> Stream")
            with _scoped_library("test_return_stream", "IMPL", "CPU") as l_impl:
                l_impl.impl(
                    "return_stream",
                    lambda _: torch.Stream(stream_id=0, device_index=1, device_type=2),
                )

                class TestMode(TorchDispatchMode):
                    # [CLASS DOC] HELPER: Dispatch mode that returns a Stream with id 1, device
                    # index 2 and device type 3 for every op.
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        # [FUNCTION DOC] HELPER: Returns a fixed Stream object.
                        return torch.Stream(stream_id=1, device_index=2, device_type=3)

                t = torch.tensor(5.0)
                s = torch.ops.test_return_stream.return_stream(t)
                self.assertIsInstance(s, torch.Stream)
                self.assertEqual(s.stream_id, 0)
                self.assertEqual(s.device_index, 1)
                self.assertEqual(s.device_type, 2)

                with TestMode():
                    s = torch.ops.test_return_stream.return_stream(t)
                self.assertIsInstance(s, torch.Stream)
                self.assertEqual(s.stream_id, 1)
                self.assertEqual(s.device_index, 2)
                self.assertEqual(s.device_type, 3)

    def test_none_wrapping(self):
        # [FUNCTION DOC] Uses a wrapper subclass whose dispatch hook returns None for add.
        # Checks multiplication still yields the subclass, add yields None, and that backward
        # through acos (which adds internally) raises RuntimeError containing but got None.
        # A Tensor subclass that returns None when doing add
        # See LoggingTensor above for more details on the subclass
        class SubclassWithNone(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass that unwraps and re-wraps tensors around ops
            # but returns None for the add op.
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the wrapper through the keyword
                # make-wrapper-subclass hook and stores elem.
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the op on unwrapped inputs, re-wraps outputs, then
                # returns None if the op is add and the wrapped result otherwise.
                def unwrap(e):
                    # [FUNCTION DOC] HELPER: Returns the inner tensor of a SubclassWithNone,
                    # else the value as is.
                    return e.elem if isinstance(e, SubclassWithNone) else e

                def wrap(e):
                    # [FUNCTION DOC] HELPER: Wraps tensors into SubclassWithNone and leaves
                    # other values unchanged.
                    return SubclassWithNone(e) if isinstance(e, torch.Tensor) else e

                rs = tree_map(
                    wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
                )
                if func.overloadpacket.__name__ == "add":
                    return None
                else:
                    return rs

        x = SubclassWithNone(torch.rand(2))
        # Make sure both run without error
        self.assertIsInstance(x * 2, SubclassWithNone)
        self.assertIsNone(x + 2)

        x.requires_grad_()
        out = x.acos().sum()

        # The backward of acos does add then rsqrt so here we make sure that the
        # undefined Tensor generated by the user code is nicely handled.
        # If acos formula changes in the future, this can be replaced by any other
        # function that does add then something in the backward in a composite way
        with self.assertRaisesRegex(RuntimeError, "but got None"):
            out.backward()

    def test_storage_can_be_converted_to_python_object(self):
        # [FUNCTION DOC] Creates an empty storage object and sets it as the storage of a
        # LoggingTensor, checking the storage can be converted to a Python object without error.
        s = torch.Storage()
        z = LoggingTensor(torch.empty([]))
        z.set_(s)

    def test_autograd_in_attr(self):
        # [FUNCTION DOC] Wraps a requires-grad tensor in a reentrant logging subclass and adds a
        # constant. Checks the outer result does not require grad while the inner one does with
        # a grad function, backward on the outer sum raises RuntimeError about not requiring
        # grad, and backward on the inner populates the inner gradient but not the outer one.
        # We want the wrapped Tensor to require gradients!
        true_t = torch.rand(2, requires_grad=True)
        t = LoggingTensorReentrant(true_t)

        out = t + 2

        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)

        self.assertTrue(out.elem.requires_grad)
        self.assertIsNotNone(out.elem.grad_fn)

        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            out.sum().backward()

        out.elem.sum().backward()

        self.assertIsNone(t.grad)
        self.assertIsNotNone(t.elem.grad)

    def test_dispatch_super_call(self):
        # [FUNCTION DOC] Checks that a subclass hook can delegate to the base dispatch
        # implementation: adding two subclass instances gives the plain result and the hook saw
        # exactly one call, to the Tensor overload of add.
        called = []

        class SubTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass created with the make-subclass helper whose
            # dispatch hook records the op in an enclosing list and delegates to the base class
            # hook.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem.
                return torch.Tensor._make_subclass(cls, elem)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Records the op and returns the result of the base class
                # dispatch hook.
                called.append(func)
                return super().__torch_dispatch__(func, types, args, kwargs)

        x = torch.randn(2)
        y = torch.randn(2)
        self.assertEqual(SubTensor(x) + SubTensor(y), x + y)
        self.assertEqual(called, [torch.ops.aten.add.Tensor])

    def test_dispatch_super_call_list_arg(self):
        # [FUNCTION DOC] Same as the super-call test but the hook passes the positional args to
        # the base hook as a list instead of a tuple; checks neg gives the plain result and one
        # call is recorded.
        called = []

        class SubTensorWithListArg(torch.Tensor):
            # [CLASS DOC] HELPER: Subclass whose dispatch hook records the op and calls the base
            # hook with args converted to a list.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem.
                return torch.Tensor._make_subclass(cls, elem)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Records the op and delegates to the base hook with a
                # list of args.
                called.append(func)
                return super().__torch_dispatch__(func, types, list(args), kwargs)

        x = torch.randn(2)
        self.assertEqual(SubTensorWithListArg(x).neg(), x.neg())
        self.assertEqual(called, [torch.ops.aten.neg.default])

    def test_dispatch_super_dont_autograd(self):
        # [FUNCTION DOC] Checks that the base dispatch hook does not run autograd again: inside
        # the hook the input still requires grad, but the result of the base hook must not.
        called = []

        class SubTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Subclass that preserves requires_grad and whose hook asserts
            # the input requires grad and the base hook result does not.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem, preserving
                # requires_grad.
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Records the op, asserts the first argument requires
                # grad, calls the base hook, asserts the result does not require grad, and
                # returns it.
                called.append(func)
                # This argument still requires grad because it was passed
                # through directly...
                self.assertTrue(args[0].requires_grad)
                r = super().__torch_dispatch__(func, types, args, kwargs)
                # But the output better not require grad, because that means
                # you did autograd again in torch dispatch (oops)
                self.assertFalse(r.requires_grad)
                return r

        x = SubTensor(torch.randn(2, requires_grad=True))
        x.neg()
        self.assertEqual(called, [torch.ops.aten.neg.default])

    def test_set_data(self):
        # [FUNCTION DOC] Counts hook invocations while accessing the data attribute, assigning
        # to it and calling in-place set on a subclass. Reading data and set each dispatch once,
        # assigning data does not dispatch, and the type stays the subclass throughout.
        called = 0

        class SubTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Subclass whose hook increments an enclosing counter and
            # delegates to the base hook.
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Increments the counter and returns the base hook
                # result.
                nonlocal called
                called += 1
                return super().__torch_dispatch__(func, types, args, kwargs)

        x = SubTensor(torch.empty(2))
        x.data
        self.assertEqual(called, 1)
        x.data = torch.empty(2)
        self.assertEqual(called, 1)
        x.data
        self.assertEqual(called, 2)
        self.assertIs(type(x), SubTensor)
        x.set_(torch.empty(2))
        self.assertEqual(called, 3)
        x.data
        self.assertEqual(called, 4)
        self.assertIs(type(x), SubTensor)

    def test_construct_int_tensor(self):
        # [FUNCTION DOC] Constructs a plain Tensor subclass from an int32 tensor and expects no
        # failure.
        class SubTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass.
            pass

        # should not fail
        SubTensor(torch.zeros(2, dtype=torch.int))

    def test_multiple_ops_subclass(self):
        # [FUNCTION DOC] Regression test for a TLS handling bug: a subclass whose hook runs the
        # op with dispatch disabled gets a conjugate view of a complex tensor, then calls exp,
        # which makes the dispatcher call clone through the conjugate fallback. The test only
        # requires that this does not raise.
        # This is a Direct Subclass, don't do that!
        class MySubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Direct Tensor subclass whose hook runs every op with dispatch
            # disabled.
            @staticmethod
            def __new__(cls, elem):
                # [FUNCTION DOC] HELPER: Creates the subclass instance from elem with the
                # make-subclass helper.
                r = torch.Tensor._make_subclass(cls, elem)
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the op with dispatch disabled and returns its
                # result.
                with no_dispatch():
                    return func(*args, **kwargs)

        x = MySubclass(torch.rand(2, 2, dtype=torch.complex64))
        y = x.conj()
        # Details of the bug that this tests for:
        # Here, y dispatch keys are: {PythonTLSSnapshot, AutogradCPU, Conjugate, Python, CPU}
        # There are a few calls to the dispatcher that are going to happen here:
        #  - call_exp: User calling exp on y
        #    - PythonTLSSnapshot: records the TLS on entry and redispatch
        #    - AutogradCPU: no input requires grad, so does nothing and redispatch
        #    - Conjugate: no special implementation for exp: use the fallback that
        #                 first clone the Tensor (to materialize the conj) then redispatch
        #      - call_clone: conjugate fallback calling clone on y
        #        - PythonTLSSnapshot: records the TLS on entry and redispatch
        #        - (AutogradCPU: skipped as autograd added itself to the exclude set above)
        #        - Conjugate: special implementation for clone: just skip this key
        #        - Python: Reset the TLS based on the snapshot above and call the user implementation (this
        #                  actually calls into the dispatcher again but since we disable both our keys
        #                  before, not detailed here)
        #        - exit Python: restore the TLS and exit
        #        - exit Conjugate: nothing was inplace so just exit
        #        - exit PythonTLSSnapshot: done with this call, reset the saved TLS to empty
        #    - Python: Reset the TLS again based on the snapshot. <- this used to fail
        #    - More steps....
        y.exp()

    @staticmethod
    def subclass_helper(cls, data, use_wrapper_subclass, **kwargs):
        # [FUNCTION DOC] HELPER: Static helper shared by the sizes/strides tests. Builds either
        # a wrapper subclass (copying device and dtype and layout from data, requires_grad
        # forced on, extra kwargs passed through, size from data) or a plain make-subclass
        # tensor with requires_grad on and extra kwargs.
        if use_wrapper_subclass:
            kwargs["device"] = data.device
            kwargs["dtype"] = data.dtype
            kwargs["layout"] = data.layout
            kwargs["requires_grad"] = True
            return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)  # type: ignore[attr-defined]
        else:
            return torch.Tensor._make_subclass(cls, data, True, **kwargs)

    def test_is_contiguous_slow_path(self):
        # [FUNCTION DOC] For both wrapper and non-wrapper subclasses with the strides dispatch
        # policy, checks is_contiguous: a hook returning NotImplemented raises TypeError (also
        # from contiguous), a hook reporting contiguous returns true and contiguous returns the
        # original, and a hook reporting non-contiguous gives false and contiguous raises
        # TypeError.
        data = torch.randn(3, 3)
        contiguous_data = data.clone()
        not_contiguous_data = torch.as_strided(data.clone(), (2, 2), (1, 2))

        for use_wrapper_subclass in [True, False]:

            class ExampleTensor1(torch.Tensor):
                # [CLASS DOC] HELPER: Subclass using the strides policy whose hook declines
                # every op.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the strides policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns NotImplemented for every op.
                    return NotImplemented

            class ExampleTensor2(torch.Tensor):
                # [CLASS DOC] HELPER: Strides-policy subclass whose hook answers is_contiguous
                # and sym_is_contiguous with the answer for an enclosing contiguous tensor,
                # declining other ops.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the strides policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Answers is_contiguous and the symbolic variant from
                    # the contiguous reference tensor, otherwise returns NotImplemented.
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return contiguous_data.is_contiguous()
                    if func.overloadpacket == torch.ops.aten.sym_is_contiguous:
                        return torch.ops.aten.sym_is_contiguous(contiguous_data)
                    return NotImplemented

            class ExampleTensor3(torch.Tensor):
                # [CLASS DOC] HELPER: Like the previous class but answers using a non-contiguous
                # reference tensor.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the strides policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Answers is_contiguous and the symbolic variant from
                    # the non-contiguous reference tensor, otherwise returns NotImplemented.
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return not_contiguous_data.is_contiguous()
                    if func.overloadpacket == torch.ops.aten.sym_is_contiguous:
                        return torch.ops.aten.sym_is_contiguous(not_contiguous_data)
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.aten.is_contiguous'"
            e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.is_contiguous()
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()

            e = ExampleTensor2(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), True)
            e.contiguous()  # this will just return the original TensorImpl since is_contiguous = True

            err_msg = "Multiple dispatch failed for"
            e = ExampleTensor3(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), False)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()

    def test_fancy_strides(self):
        # [FUNCTION DOC] Uses a non-wrapper subclass with the strides policy whose hook records
        # and answers None for contiguity, stride and layout queries, and runs other ops with
        # dispatch disabled. Checks the channels_last contiguity check, strides-like-format and
        # non-overlapping-and-dense queries each reach the hook with the expected extra args.
        calls = []

        class ExampleTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Strides-policy subclass whose hook logs a listed set of
            # metadata queries to an enclosing list (returning None) and executes everything
            # else with dispatch disabled.
            @staticmethod
            def __new__(cls, data):
                # [FUNCTION DOC] HELPER: Creates a non-wrapper subclass through the shared
                # helper with the strides policy.
                return TestPythonDispatch.subclass_helper(
                    cls, data, False, dispatch_sizes_strides_policy="strides"
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: For the listed metadata ops, appends the op and the
                # args after the first to the log and returns None; otherwise runs the op with
                # dispatch disabled.
                if func in [
                    torch.ops.aten.sym_is_contiguous.default,
                    torch.ops.aten.is_contiguous.default,
                    torch.ops.aten.is_contiguous.memory_format,
                    torch.ops.aten.is_strides_like_format.default,
                    torch.ops.aten.is_non_overlapping_and_dense.default,
                    torch.ops.aten.stride.default,
                ]:
                    calls.append((func, list(args)[1:]))
                    return None
                with no_dispatch():
                    return func(*args, **kwargs)

        e = ExampleTensor(torch.randn(2, 2))
        self.assertFalse(e.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(
            calls, [(torch.ops.aten.is_contiguous.memory_format, [torch.channels_last])]
        )
        calls.clear()
        self.assertFalse(
            torch.ops.aten.is_strides_like_format.default(e, torch.channels_last)
        )
        self.assertEqual(
            calls,
            [(torch.ops.aten.is_strides_like_format.default, [torch.channels_last])],
        )
        calls.clear()
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(e))
        self.assertEqual(
            calls, [(torch.ops.aten.is_non_overlapping_and_dense.default, [])]
        )

    def test_device_slowpath(self):
        # [FUNCTION DOC] With a wrapper subclass using device dispatch: a hook declining the
        # device query makes accessing it raise TypeError; hooks returning the meta device make
        # the tensor report device type meta, and type_as on another tensor also reports meta.
        for use_wrapper_subclass in [True]:

            class ExampleTensor1(torch.Tensor):
                # [CLASS DOC] HELPER: Device-dispatching subclass whose hook declines every op.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # device dispatch on.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns NotImplemented for every op.
                    return NotImplemented

            class ExampleTensor2(torch.Tensor):
                # [CLASS DOC] HELPER: Device-dispatching subclass whose hook reports the meta
                # device for the device query.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # device dispatch on.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns the meta device for the prim device op,
                    # otherwise NotImplemented.
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device("meta")
                    return NotImplemented

            class ExampleTensor3(torch.Tensor):
                # [CLASS DOC] HELPER: Same behavior as the previous class (meta device for the
                # device query); a second variant for the same assertions.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # device dispatch on.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns the meta device for the prim device op,
                    # otherwise NotImplemented.
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device("meta")
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.prim.device'"
            with self.assertRaisesRegex(TypeError, err_msg):
                e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
                e.device()

            ten = torch.rand([1])
            e = ExampleTensor2(torch.randn(3, 3, device="cpu"), use_wrapper_subclass)
            self.assertEqual(e.device.type, "meta")
            self.assertEqual(ten.type_as(e).device.type, "meta")

            e = ExampleTensor3(torch.randn(3, 3, device="cpu"), use_wrapper_subclass)
            self.assertEqual(e.device.type, "meta")
            self.assertEqual(ten.type_as(e).device.type, "meta")

    def test_dim_slowpath(self):
        # [FUNCTION DOC] For wrapper and non-wrapper subclasses with the sizes policy: a hook
        # declining everything makes calling dim raise TypeError; a hook answering dim from the
        # data returns 2.
        data = torch.randn(3, 3)

        for use_wrapper_subclass in [True, False]:

            class DimNotImplementedTensor(torch.Tensor):
                # [CLASS DOC] HELPER: Sizes-policy subclass whose hook declines every op.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the sizes policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns NotImplemented for every op.
                    return NotImplemented

            class DimImplementedTensor(torch.Tensor):
                # [CLASS DOC] HELPER: Sizes-policy subclass whose hook answers dim from the
                # enclosing data tensor.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the sizes policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns the dimension count of the data for the dim
                    # op, otherwise NotImplemented.
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.aten.dim'"
            e = DimNotImplementedTensor(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.dim()

            t = DimImplementedTensor(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(t.dim(), 2)

    def test_maybe_tuple_bug(self):
        # [FUNCTION DOC] Regression test: indexes a tensor with a list of two instances of a
        # class that overrides the torch function protocol and returns None; passes if no
        # exception is raised.
        class T(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass with a torch function override that does
            # nothing.
            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Torch function hook returning None for any call.
                pass

        a = torch.rand(3)

        a[[T(), T()]]

    def test_standard_is_not_subclass(self):
        # [FUNCTION DOC] Checks the private subclass-like predicate is false for a plain empty
        # tensor (issue 79079).
        # https://github.com/pytorch/pytorch/issues/79079
        self.assertFalse(torch._C._dispatch_isTensorSubclassLike(torch.empty(0)))

    def test_sym_sizes_strides_slow_path(self):
        # [FUNCTION DOC] Creates a wrapper subclass with the sizes policy whose hook answers
        # symbolic size and stride queries with a tuple holding one symbolic int built from a
        # ShapeEnv. Checks the first element of size and stride are SymInt.
        class TestTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass of size zero (sizes policy) whose hook
            # returns a one-element tuple with a SymInt for sym_size and sym_stride and None for
            # other ops.
            @staticmethod
            def __new__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the wrapper through the private hook with a
                # zero-length shape and the sizes policy.
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls, (0,), dispatch_sizes_strides_policy="sizes"
                )
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: For sym_size and sym_stride, builds a ShapeEnv, creates
                # a symbol with hint 123 and a constant source and returns a tuple with the
                # resulting SymInt; other ops fall through and return None.
                if func in (
                    torch.ops.aten.sym_size.default,
                    torch.ops.aten.sym_stride.default,
                ):
                    from torch._dynamo.source import ConstantSource
                    from torch.fx.experimental.symbolic_shapes import (
                        DimDynamic,
                        ShapeEnv,
                    )

                    shape_env = ShapeEnv()
                    si = shape_env.create_symintnode(
                        shape_env.create_symbol(
                            123,
                            source=ConstantSource("abc"),
                            dynamic_dim=DimDynamic.DUCK,
                            constraint_dim=None,
                        ),
                        hint=123,
                    )
                    return (si,)

        t = TestTensor()
        si = t.size()[0]
        self.assertIsInstance(si, torch.SymInt)
        si = t.stride()[0]
        self.assertIsInstance(si, torch.SymInt)

    def test_strides_slow_path(self):
        # [FUNCTION DOC] For wrapper and non-wrapper subclasses with the strides policy:
        # declining the symbolic stride raises TypeError; a hook returning a custom tuple four
        # and two is used for stride; a hook returning None falls back to the real strides two
        # and one.
        for use_wrapper_subclass in [True, False]:

            class StridesNotImplemented(torch.Tensor):
                # [CLASS DOC] HELPER: Strides-policy subclass whose hook declines every op.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the strides policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns NotImplemented for every op.
                    return NotImplemented

            class StridesCustomReturn(torch.Tensor):
                # [CLASS DOC] HELPER: Strides-policy subclass whose hook returns a fixed stride
                # tuple for the symbolic stride op.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the strides policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns the tuple four and two for the symbolic
                    # stride op, otherwise NotImplemented.
                    if func == torch.ops.aten.sym_stride.default:
                        return (4, 2)
                    return NotImplemented

            class StridesDefaultReturn(torch.Tensor):
                # [CLASS DOC] HELPER: Strides-policy subclass whose hook returns None for the
                # symbolic stride op, meaning use the default.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the strides policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns None for the symbolic stride op, otherwise
                    # NotImplemented.
                    if func == torch.ops.aten.sym_stride.default:
                        return None
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.aten.sym_stride'"
            e = StridesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.stride()

            e = StridesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.stride(), (4, 2))

            e = StridesDefaultReturn(torch.randn(6, 2), use_wrapper_subclass)
            self.assertEqual(e.stride(), (2, 1))

    def test_sizes_slow_path(self):
        # [FUNCTION DOC] For wrapper and non-wrapper subclasses with the sizes policy: declining
        # sym_size makes size raise TypeError; a hook returning five by three is used for size;
        # a hook returning None falls back to the real shape four and two.
        for use_wrapper_subclass in [True, False]:
            data = torch.randn(6, 2)

            class SizesNotImplemented(torch.Tensor):
                # [CLASS DOC] HELPER: Sizes-policy subclass whose hook answers only dim (from
                # the enclosing data) and declines the rest, including sym_size.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the sizes policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns the dimension count for dim, else
                    # NotImplemented.
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    return NotImplemented

            class SizesCustomReturn(torch.Tensor):
                # [CLASS DOC] HELPER: Sizes-policy subclass whose hook answers dim and returns a
                # fixed size tuple five by three for sym_size.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the sizes policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns dimension count for dim, the tuple five by
                    # three for sym_size, else NotImplemented.
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return (5, 3)
                    return NotImplemented

            class SizesDefaultReturn(torch.Tensor):
                # [CLASS DOC] HELPER: Sizes-policy subclass whose hook answers dim and returns
                # None for sym_size to use the default.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # the sizes policy.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns dimension count for dim, None for sym_size,
                    # else NotImplemented.
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return None
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.aten.sym_size'"
            e = SizesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.size()

            e = SizesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.size(), (5, 3))

            e = SizesDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.size(), (4, 2))

    def test_custom_size_policy_dynamic_shapes(self):
        # [FUNCTION DOC] Traces a function creating a custom wrapper subclass with the sizes
        # policy and querying its size and stride, using make_fx with symbolic tracing. Compares
        # the generated graph code with the expected text, where both size and stride come out
        # as the symbolic sizes of the input.
        data = torch.randn(6, 2)

        class CustomSizeDynamicShapesTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper subclass storing an inner tensor in an attribute; its
            # hook answers symbolic size and stride queries with the inner tensor shape,
            # declining others.
            @staticmethod
            def __new__(cls, inner):
                # [FUNCTION DOC] HELPER: Creates the wrapper through the positional overload of
                # the private hook with the inner size, stride, dtype, layout, device,
                # requires_grad and the sizes policy.
                return torch.Tensor._make_wrapper_subclass(
                    # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
                    # Calling the overload that has kwargs causes us to go down the first overload path,
                    # which will **always** specialize sizes.
                    # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
                    cls,
                    inner.size(),
                    inner.stride(),
                    None,
                    None,
                    inner.dtype,
                    inner.layout,
                    inner.device,
                    False,
                    inner.requires_grad,
                    "sizes",
                )

            def __init__(self, inner):
                # [FUNCTION DOC] HELPER: Stores the inner tensor in an attribute.
                self.inner = inner

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Returns the inner tensor shape for the symbolic size
                # and stride ops (even for stride, deliberately), else NotImplemented.
                if func == torch.ops.aten.sym_size.default:
                    return args[0].inner.shape
                if func == torch.ops.aten.sym_stride.default:
                    return args[0].inner.shape
                return NotImplemented

        x = torch.ones(2, 2)

        def trace_fn(x):
            # [FUNCTION DOC] HELPER: Function being traced: wraps x in the custom subclass and
            # returns the pair of its size and stride.
            x_wrapper = CustomSizeDynamicShapesTensor(x)
            return x_wrapper.size(), x_wrapper.stride()

        fx_g = make_fx(trace_fn, tracing_mode="symbolic")(x)
        self.assertExpectedInline(
            fx_g.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1);  x_1 = None
    return ((sym_size_int, sym_size_int_1), (sym_size_int, sym_size_int_1))""",
        )

    def test_data_ptr_respects_numel_slow_path(self):
        # [FUNCTION DOC] For both subclass flavors with the sizes policy, calls the data pointer
        # method and checks the hook was asked for numel (the flag in the enclosing list becomes
        # true).
        data = torch.randn(6, 2)

        class NumelDefaultReturn(torch.Tensor):
            # [CLASS DOC] HELPER: Sizes-policy subclass whose hook answers dim, returns None for
            # numel while setting a flag in an enclosing list, and declines the rest.
            @staticmethod
            def __new__(cls, data, wrapper):
                # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with the
                # sizes policy.
                return TestPythonDispatch.subclass_helper(
                    cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Returns dimension count for dim; for numel sets the
                # flag and returns None; otherwise NotImplemented.
                if func.overloadpacket == torch.ops.aten.dim:
                    return data.dim()
                if func.overloadpacket == torch.ops.aten.numel:
                    numel_called[0] = True
                    return None
                return NotImplemented

        for use_wrapper_subclass in (False, True):
            numel_called = [False]
            e = NumelDefaultReturn(torch.randn(2, 2), use_wrapper_subclass)
            e.data_ptr()
            self.assertTrue(numel_called[0])

    def test_layout_slow_path(self):
        # [FUNCTION DOC] For wrapper and non-wrapper subclasses with layout dispatch: a
        # declining hook makes layout access raise TypeError; a hook returning sparse CSR is
        # reflected; a hook returning the data layout gives strided.
        for use_wrapper_subclass in [True, False]:
            data = torch.randn(6, 2)

            class LayoutNotImplemented(torch.Tensor):
                # [CLASS DOC] HELPER: Layout-dispatching subclass whose hook declines every op.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # layout dispatch on.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns NotImplemented for every op.
                    return NotImplemented

            class LayoutCustomReturn(torch.Tensor):
                # [CLASS DOC] HELPER: Layout-dispatching subclass whose hook reports the sparse
                # CSR layout.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # layout dispatch on.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns sparse CSR for the prim layout op,
                    # otherwise NotImplemented.
                    if func.overloadpacket == torch.ops.prim.layout:
                        return torch.sparse_csr
                    return NotImplemented

            class LayoutDefaultReturn(torch.Tensor):
                # [CLASS DOC] HELPER: Layout-dispatching subclass whose hook reports the layout
                # of the enclosing data tensor.
                @staticmethod
                def __new__(cls, data, wrapper):
                    # [FUNCTION DOC] HELPER: Creates the subclass through the shared helper with
                    # layout dispatch on.
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # [FUNCTION DOC] HELPER: Returns the data layout for the prim layout op,
                    # otherwise NotImplemented.
                    if func.overloadpacket == torch.ops.prim.layout:
                        return data.layout
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.prim.layout'"
            e = LayoutNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.layout

            e = LayoutCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.sparse_csr)

            e = LayoutDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.strided)

    def test_wrapper_subclass_reentrant_dispatch_with_mode(self):
        # [FUNCTION DOC] Regression test for issue 136565: a TwoTensor-derived subclass hook
        # re-dispatches on its first inner tensor using reentrant dispatch, while a pass-through
        # dispatch mode is active. Calling clone must return a plain tensor equal to the inner
        # tensor.
        # Tests the interaction between a wrapper subclass using reentrant dispatch
        # and a TorchDispatchMode. See https://github.com/pytorch/pytorch/issues/136565

        # simple passthrough TorchDispatchMode
        class CustomDispatchMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Pass-through dispatch mode.
            def __torch_dispatch__(self, func, types, args=..., kwargs=None):
                # [FUNCTION DOC] HELPER: Runs the op with the given args and kwargs.
                return func(*args, **kwargs)

        # derive from TwoTensor to minimize boilerplate
        class MySubclass(TwoTensor):
            # [CLASS DOC] HELPER: TwoTensor subclass whose hook re-dispatches the op on the
            # first inner tensor under the reentrant dispatch context.
            def __torch_dispatch__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Under reentrant dispatch, runs the op on the first
                # inner tensor of the first argument only (ignores other args) and returns the
                # plain result.
                with torch.overrides.enable_reentrant_dispatch():
                    return func(args[0].a)

        t = MySubclass(torch.rand(2), torch.rand(2))
        with CustomDispatchMode():
            res = t.clone()

        self.assertEqual(res, t.a)
        self.assertIs(type(res), torch.Tensor)

    def test_custom_dispatch_mode_supports_higher_order_operators(self):
        # [FUNCTION DOC] Checks that a mode declaring support for higher-order operators
        # receives a cond call and its returned ones tensor becomes the result.
        class Mode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Mode with higher-order operator support enabled that returns a
            # tensor of ones for the cond higher-order op and declines others.
            supports_higher_order_operators = True

            def __torch_dispatch__(self, func, types, args=..., kwargs=None):
                # [FUNCTION DOC] HELPER: Returns a 3 by 3 ones tensor for the cond op, otherwise
                # NotImplemented.
                if func is torch.ops.higher_order.cond:
                    return torch.ones(3, 3)
                return NotImplemented

        pred = torch.tensor(True)
        x = torch.randn(1, 1)
        with Mode():
            out = torch.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))
        self.assertEqual(out, torch.ones(3, 3))

    def test_custom_dispatch_mode_not_supports_higher_order_operators(self):
        # [FUNCTION DOC] Checks that a mode not supporting higher-order operators leads cond to
        # raise NotImplementedError saying no rule was registered for the operator and mode.
        class Mode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Mode with higher-order operator support disabled that would
            # return ones for cond if it were called.
            supports_higher_order_operators = False

            def __torch_dispatch__(self, func, types, args=..., kwargs=None):
                # [FUNCTION DOC] HELPER: Returns a 3 by 3 ones tensor for the cond op, otherwise
                # NotImplemented; never reached in the test.
                if func is torch.ops.higher_order.cond:
                    return torch.ones(3, 3)
                return NotImplemented

        pred = torch.tensor(True)
        x = torch.randn(1, 1)
        with self.assertRaisesRegex(
            NotImplementedError,
            "There was no rule registered for HigherOrderOperator cond and mode",
        ):
            with Mode():
                torch.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))

    def test_dispatch_uint64(self):
        # [FUNCTION DOC] Creates a uint64 tensor filled with a value above the signed int64
        # range under a recording mode; checks the fill value reaches the hook unchanged and the
        # tensor equals that value everywhere.
        class DummyMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Pass-through dispatch mode that remembers the last args it saw
            # in an attribute named last_args (assigned in the hook, not declared up front).
            def __torch_dispatch__(self, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Stores args as last_args, then runs the op and returns
                # the result.
                self.last_args = args
                return func(*args, **kwargs)

        # Value that could not be interpreted as signed int64
        uarg = 2**63 + 1
        with DummyMode() as m:
            a = torch.full((3, 3), uarg, dtype=torch.uint64)
            self.assertEqual(m.last_args[1], uarg)
        self.assertTrue((a == uarg).all().item())


class TestPythonDispatcher(TestCase):
    # [CLASS DOC] Tests running ops with the Python dispatcher enabled via the private enable
    # helper.
    def test_basic(self):
        # [FUNCTION DOC] Enables the Python dispatcher and runs add on a requires-grad tensor;
        # passes if nothing raises.
        x = torch.randn(2, requires_grad=True)
        r = torch._C._EnablePythonDispatcher()
        torch.add(x, x)

    def test_lstsq(self):
        # [FUNCTION DOC] Computes the shape of a least-squares solution with and without the
        # Python dispatcher enabled and checks the shapes match.
        a = torch.randn(4, 3)
        b = torch.rand(4, 3)
        expected_shape = torch.linalg.lstsq(a, b).solution.shape
        r = torch._C._EnablePythonDispatcher()
        python_disp_shape = torch.linalg.lstsq(a, b).solution.shape
        self.assertEqual(expected_shape, python_disp_shape)


class TestWrapperSubclassAliasing(TestCase):
    # [CLASS DOC] Device-generic tests (instantiated per device below) that wrapper subclasses
    # via the aliasing correction utility preserve input/output aliasing like plain tensors.
    # Uses TwoTensor and compares storage identity of outputs and inputs between plain and
    # subclass runs.
    def _test_wrapper_subclass_aliasing(self, op, args, kwargs):
        # [FUNCTION DOC] HELPER: Runs op on plain args, then on args with every tensor wrapped
        # in a TwoTensor. For each pair of output tensor and input tensor, checks: output
        # identical to an input implies the same for the subclass run; output sharing storage
        # with an input implies the subclass outputs share storage with the matching input,
        # otherwise they must not.
        def to_subclass(t: torch.Tensor):
            # [FUNCTION DOC] HELPER: Wraps a tensor in a TwoTensor of itself and a clone.
            return TwoTensor(t, t.clone())

        result_ref = op(*args, **kwargs)

        args_subclass = pytree.tree_map_only(torch.Tensor, to_subclass, args)
        kwargs_subclass = pytree.tree_map_only(torch.Tensor, to_subclass, kwargs)

        result_test = op(*args_subclass, **kwargs_subclass)

        args_ref_flat = pytree.arg_tree_leaves(*args, **kwargs)
        args_ref_flat_tensors = [
            x for x in args_ref_flat if isinstance(x, torch.Tensor)
        ]

        args_test_flat = pytree.tree_leaves((args_subclass, kwargs_subclass))
        args_test_flat_tensors = [
            x for x in args_test_flat if isinstance(x, torch.Tensor)
        ]

        result_ref_flat = pytree.tree_leaves(result_ref)
        result_ref_flat_tensors = [
            x for x in result_ref_flat if isinstance(x, torch.Tensor)
        ]

        result_test_flat = pytree.tree_leaves(result_test)
        result_test_flat_tensors = [
            x for x in result_test_flat if isinstance(x, torch.Tensor)
        ]

        for o_ref, o_test in zip(result_ref_flat_tensors, result_test_flat_tensors):
            for a_ref, a_test in zip(args_ref_flat_tensors, args_test_flat_tensors):
                out_is_inpt = o_ref is a_ref
                if out_is_inpt:
                    self.assertTrue(o_test is a_test)

                out_aliases_inpt = StorageWeakRef(
                    o_ref.untyped_storage()
                ) == StorageWeakRef(a_ref.untyped_storage())
                if out_aliases_inpt:
                    self.assertTrue(
                        StorageWeakRef(o_test.untyped_storage())
                        == StorageWeakRef(a_test.untyped_storage())
                    )
                else:
                    self.assertFalse(
                        StorageWeakRef(o_test.untyped_storage())
                        == StorageWeakRef(a_test.untyped_storage())
                    )

    # This tests the correctness of `torch.utils._python_dispatch.return_and_correct_aliasing`,
    # a util for wrapper subclasses to promise correct aliasing behavior.
    # It's probably overkill to test every OpInfo,
    # so I picked a sampling of ops with representative schemas.
    @ops(
        [
            op
            for op in op_db
            if op.name
            in [
                "mul",  # out-of-place
                "cat",  # out-of-place (TensorList input)
                "index",  # out-of-place (Optional TensorList input)
                "mul_",  # inplace
                "view",  # view
                "t_",  # inplace-view
                "split",  # view (multi-return)
                "native_batch_norm",  # mutable op (returns outputs and mutates some inputs)
            ]
        ],
        allowed_dtypes=(torch.float,),
    )
    def test_wrapper_subclass_aliasing(self, device, dtype, op):
        # [FUNCTION DOC] Parametrized over a selection of OpInfos (mul, cat, index, mul_, view,
        # t_, split, native_batch_norm) restricted to float dtype. Takes the first sample input
        # for the op and checks aliasing via the shared helper.
        samples = op.sample_inputs(device, dtype)
        sample = first_sample(self, samples)
        args = (sample.input, *sample.args)
        kwargs = sample.kwargs
        self._test_wrapper_subclass_aliasing(op, args, kwargs)

    @ops(custom_op_db, allowed_dtypes=(torch.float,))
    def test_wrapper_subclass_aliasing_custom(self, device, dtype, op):
        # [FUNCTION DOC] Same as the previous test but over the custom op database with float
        # dtype.
        samples = op.sample_inputs(device, dtype)
        sample = first_sample(self, samples)
        args = (sample.input, *sample.args)
        kwargs = sample.kwargs
        self._test_wrapper_subclass_aliasing(op, args, kwargs)

    def test_wrapper_subclass_aliasing_conv2d(self, device):
        # [FUNCTION DOC] Runs the aliasing check on the conv2d default overload with inference
        # mode so it reaches dispatch undecomposed, covering default list args expanded by the
        # schema.
        args = (
            torch.randn(4, 4, 4, 4, device=device),
            torch.randn(4, 4, 4, 4, device=device),
        )
        kwargs = {}
        # conv2d has a default arg 'int[2] strides=0',
        # which torchscript expands into 'int[2] strides=[0, 0]'
        # Make sure that _return_and_correct_aliasing can handle this case
        # (I'm using inference_mode to make sure conv2d doesn't decompose and goes to torch_dispatch)
        with torch.inference_mode():
            self._test_wrapper_subclass_aliasing(
                torch.ops.aten.conv2d.default, args, kwargs
            )

    def test_wrapper_subclass_aliasing_out_op(self, device):
        # [FUNCTION DOC] Runs the aliasing check on the add out overload, where a mutable tensor
        # is passed as a keyword argument.
        # Make sure that _return_and_correct_aliasing can handle kwargs w mutable tensors
        args = (torch.ones(4, device=device), torch.ones(4, device=device))
        kwargs = {"out": torch.empty(4, device=device)}
        self._test_wrapper_subclass_aliasing(torch.ops.aten.add.out, args, kwargs)

    def test_wrapper_subclass_aliasing_fft_fft2(self, device):
        # [FUNCTION DOC] Runs the aliasing check on fft2 under inference mode, covering a
        # default list dimension argument.
        args = (torch.randn(4, 4, device=device),)
        kwargs = {}
        # fft_fft2 has a default arg 'int[1] dim=[-2,-1]',
        # Make sure that _return_and_correct_aliasing can handle this case
        # (I'm using inference_mode to make sure fft_fft2 doesn't decompose and goes to torch_dispatch)
        with torch.inference_mode():
            self._test_wrapper_subclass_aliasing(torch.ops.aten.fft_fft2, args, kwargs)


instantiate_device_type_tests(TestWrapperSubclassAliasing, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
