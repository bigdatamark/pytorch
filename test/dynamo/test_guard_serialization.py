# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for serializing dynamo guards (save_guards / load_guard_manager) and
# checking that the reloaded guard manager agrees with the original one. Defines helper modules,
# tensor subclasses and a custom pytree constant type at module scope (guard serialization needs
# globally importable classes), plus a base test case that traces a function, builds guards and
# round-trips them, and the TestGuardSerialization suite with one test per guard type. Import
# time side effect: registers CustomConstantType as a constant opaque custom class.

import dataclasses
import itertools
import pickle
import sys
import tempfile
import types
import unittest
import weakref
from collections.abc import Iterator
from typing import NamedTuple

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.fx.graph as fx_graph
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.bytecode_transformation import transform_code_object
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import CheckFunctionManager, CompileId
from torch._dynamo.package import CompilePackage
from torch._dynamo.source import LocalSource
from torch._dynamo.symbolic_convert import (
    ExceptionStack,
    InstructionTranslator,
    SpeculationLog,
)
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._guards import compile_context, CompileContext, tracing
from torch.overrides import TorchFunctionMode
from torch.testing._internal.common_utils import (
    IS_LINUX,
    IS_MACOS,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils import _pytree as pytree


@dataclasses.dataclass
class _FrameState:
    # [CLASS DOC] HELPER: Plain dataclass snapshotting the first traced Python frame: its locals
    # (copied), globals, code object and builtins. Filled in by the trace function of the test
    # base class.
    f_locals: dict
    f_globals: dict
    f_code: types.CodeType
    f_builtins: dict


class GlobalModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Trivial globally defined nn.Module whose forward adds one; used where
    # a serializable, importable module type is needed.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input tensor plus one.
        return x + 1


class GlobalNestedModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Globally defined nn.Module holding a Linear(10, 10), a random 3x2
    # Parameter and a nested submodule (the given one, or a new GlobalModule when none or a
    # falsy value is given).
    def __init__(self, submodule=None):
        # [FUNCTION DOC] HELPER: Builds the module with a linear layer, a random parameter named
        # param, and the attribute nested set to the given submodule or a fresh GlobalModule.
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.param = torch.nn.Parameter(torch.randn(3, 2))
        self.nested = submodule or GlobalModule()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer to the input and adds one; the nested
        # module and param are not used.
        return self.linear(x) + 1


def global_func(x):
    # [FUNCTION DOC] HELPER: Module-level function returning its argument plus one; used to
    # trigger guards on a global function.
    return x + 1


class ModuleNotSerializable(torch.nn.Module):
    # [CLASS DOC] HELPER: nn.Module with a random Parameter whose pickling is deliberately
    # broken, to test error handling for modules that cannot be serialized.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the module with a random 3x2 Parameter named param.
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(3, 2))

    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Pickle hook that always raises NotImplementedError, making
        # instances unpicklable.
        raise NotImplementedError("not serialzable")

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input plus the parameter.
        return x + self.param


class GlobalTorchFunctionMode(TorchFunctionMode):
    # [CLASS DOC] HELPER: Pass-through TorchFunctionMode defined at module scope so it can be
    # referenced by name when serializing.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Calls the intercepted function with the original args and
        # kwargs (kwargs default to an empty mapping) and returns the result unchanged.
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class MyClass:
    # [CLASS DOC] HELPER: Plain class whose pickling raises RuntimeError; used as an
    # unserializable object.
    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Pickle hook that always raises RuntimeError saying the object
        # cannot be pickled.
        raise RuntimeError("Cannot pickle")

    def add(self, x):
        # [FUNCTION DOC] HELPER: Returns the argument plus one.
        return x + 1


class MyClassNotSerializable:
    # [CLASS DOC] HELPER: Plain class whose pickling raises NotImplementedError; unserializable
    # counterpart of MyClass.
    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Pickle hook that always raises NotImplementedError.
        raise NotImplementedError

    def add(self, x):
        # [FUNCTION DOC] HELPER: Returns the argument plus one.
        return x + 1


class Inputs:
    # [CLASS DOC] HELPER: Simple container holding an input value x and an unused value, used as
    # a structured input object in tests.
    def __init__(self, x, unused):
        # [FUNCTION DOC] HELPER: Stores the two arguments as attributes x and unused.
        self.x = x
        self.unused = unused


def _global_func_wrong_fqn(x):
    # [FUNCTION DOC] HELPER: Returns the argument plus one. Bound to a different module-level
    # name right after definition and the original name is deleted, so its qualified name does
    # not resolve to itself (wrong-fqn case).
    return x + 1


global_func_wrong_fqn = _global_func_wrong_fqn
del _global_func_wrong_fqn


class FlatModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Globally defined nn.Module whose forward adds two.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input plus two.
        return x + 2


class ModWithDict(torch.nn.Module):
    # [CLASS DOC] HELPER: nn.Module that stores an arbitrary dictionary given at construction as
    # attribute d.
    def __init__(self, d):
        # [FUNCTION DOC] HELPER: Initializes the module and stores the given dict as d.
        super().__init__()
        self.d = d


class SubclassWithMeta(torch.Tensor):
    # [CLASS DOC] HELPER: Traceable wrapper tensor subclass wrapping one inner tensor a plus an
    # arbitrary extra metadata value. Flattens to the inner tensor with extra stored in the
    # metadata mapping, so metadata participates in guards.
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] HELPER: Creates the wrapper subclass instance with size, stride,
        # offset, device, layout, requires_grad and dtype taken from a (size and stride may be
        # overridden by the optional outer arguments).
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] HELPER: Stores the inner tensor as a and the metadata value as extra.
        self.a = a
        self.extra = extra

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] HELPER: Dispatch hook: unwraps all instances in args and kwargs to
        # their inner tensors, runs the op, and if the result is a tensor rewraps it with the
        # extra of the first argument; otherwise returns the result as is.
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(SubclassWithMeta, lambda x: x.a, args)
        kwargs_a = pytree.tree_map_only(SubclassWithMeta, lambda x: x.a, kwargs)
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, torch.Tensor):
            assert isinstance(args[0], SubclassWithMeta)  # noqa: S101
            return SubclassWithMeta(out_a, extra=args[0].extra)
        return out_a

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Returns the inner tensor attribute names (just a) and a
        # metadata mapping holding extra.
        # store extra in meta
        return ["a"], {"extra": self.extra}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Rebuilds an instance from the inner tensors and metadata
        # mapping (extra is read from the metadata). Raises if meta is not a dict, or if the
        # inner tensor is a plain tensor and the outer size or stride is missing.
        assert isinstance(meta, dict)  # noqa: S101
        a = inner_tensors["a"]
        # pull out extra from meta
        extra = meta["extra"]
        if type(a) is torch.Tensor:
            assert outer_size is not None  # noqa: S101
            assert outer_stride is not None  # noqa: S101
        return SubclassWithMeta(a, extra, outer_size, outer_stride)


class SubclassWithCustomMetadataGuard(torch.Tensor):
    # [CLASS DOC] HELPER: Same as SubclassWithMeta but defines a custom metadata guard that
    # compares only the bar entry of the extra metadata, so other entries may differ without
    # failing the guard.
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] HELPER: Creates the wrapper subclass instance with geometry and dtype
        # taken from a, optionally overriding size and stride.
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] HELPER: Stores the inner tensor as a and the metadata value as extra.
        self.a = a
        self.extra = extra

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] HELPER: Dispatch hook: unwraps instances to inner tensors, runs the op,
        # and rewraps tensor results with the first argument's extra; non-tensor results are
        # returned as is.
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(
            SubclassWithCustomMetadataGuard, lambda x: x.a, args
        )
        kwargs_a = pytree.tree_map_only(
            SubclassWithCustomMetadataGuard, lambda x: x.a, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, torch.Tensor):
            assert isinstance(args[0], SubclassWithCustomMetadataGuard)  # noqa: S101
            return SubclassWithCustomMetadataGuard(out_a, extra=args[0].extra)
        return out_a

    @classmethod
    def __metadata_guard__(cls, meta1, meta2):
        # [FUNCTION DOC] HELPER: Custom metadata equivalence used by the guard: two metadata
        # mappings match iff their extra bar entries are equal. Deliberately laxer than the
        # default comparison.
        # Define custom metadata guard logic that only looks at "bar" to determine
        # metadata equivalence. This is more purposefully more lax than the default
        # guard behavior.
        return meta1["extra"]["bar"] == meta2["extra"]["bar"]

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Returns the inner tensor names (just a) and a metadata mapping
        # holding extra.
        # store extra in meta
        return ["a"], {"extra": self.extra}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Rebuilds an instance from inner tensors and metadata. Raises if
        # meta is not a dict, or if the inner tensor is a plain tensor and outer size or stride
        # is missing.
        assert isinstance(meta, dict)  # noqa: S101
        a = inner_tensors["a"]
        # pull out extra from meta
        extra = meta["extra"]
        if type(a) is torch.Tensor:
            assert outer_size is not None  # noqa: S101
            assert outer_stride is not None  # noqa: S101
        return SubclassWithCustomMetadataGuard(a, extra, outer_size, outer_stride)


class SubclassWithSubclassInnerTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Wrapper subclass whose inner tensors are the plain tensor a and a
    # nested SubclassWithMeta built from a plus one, carrying the extra metadata; flatten has no
    # metadata of its own.
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] HELPER: Creates the wrapper subclass instance with geometry and dtype
        # taken from a, optionally overriding size and stride.
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] HELPER: Stores a and creates the nested SubclassWithMeta over a plus
        # one with the given extra as inner_sub.
        self.a = a
        self.inner_sub = SubclassWithMeta(a + 1, extra=extra)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] HELPER: Dispatch hook: unwraps instances to a, runs the op, and rewraps
        # tensor results using the extra of the first argument's inner subclass; non-tensor
        # results are returned as is.
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(
            SubclassWithSubclassInnerTensor, lambda x: x.a, args
        )
        kwargs_a = pytree.tree_map_only(
            SubclassWithSubclassInnerTensor, lambda x: x.a, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, torch.Tensor):
            assert isinstance(args[0], SubclassWithSubclassInnerTensor)  # noqa: S101
            return SubclassWithSubclassInnerTensor(out_a, extra=args[0].inner_sub.extra)
        return out_a

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Returns inner tensor names a and inner_sub with no metadata.
        return ["a", "inner_sub"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Rebuilds an instance from inner tensors, taking extra from the
        # inner_sub tensor. Raises if meta is not None, or if the inner tensor is a plain tensor
        # and outer size or stride is missing.
        assert meta is None  # noqa: S101
        a = inner_tensors["a"]
        extra = inner_tensors["inner_sub"].extra
        if type(a) is torch.Tensor:
            assert outer_size is not None  # noqa: S101
            assert outer_stride is not None  # noqa: S101
        return SubclassWithSubclassInnerTensor(a, extra, outer_size, outer_stride)


# defines a custom __eq__() / __hash__() to be registered as a pytree constant type
class CustomConstantType(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Constant-like custom class with fields a and b whose equality and hash
    # look only at a (b is ignored). Registered below as a constant opaque type so it yields
    # equals-match guards.
    def __init__(self, a, b):
        # [FUNCTION DOC] HELPER: Stores the two fields.
        self.a = a
        self.b = b

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality comparing only field a; b is ignored.
        # custom eq ignores b
        return self.a == other.a

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Hash of field a only, consistent with the equality.
        # custom hash ignores b
        return hash(self.a)

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Debug string showing both fields.
        return f"CustomConstantType(a={self.a!r}, b={self.b!r})"

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: FX codegen representation: returns the constructor-style string
        # for both fields plus a mapping that makes the class name resolvable.
        return f"CustomConstantType(a={self.a!r}, b={self.b!r})", {
            "CustomConstantType": CustomConstantType
        }


torch._library.opaque_object.register_custom_class(CustomConstantType, typ="constant")


class TestGuardSerializationBase(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Base test case for guard serialization tests. Provides tracing of a function
    # to capture the first frame, a helper that builds a reference guard manager and a
    # serialized-then-loaded guard manager for one guard type, and a helper to compare their
    # results. Keeps a snapshot of fx magic methods and the default device context, restored
    # after each test. Attributes: _frame_state (captured frame or None), _cached_guards_state
    # and _cached_f_code (last serialized guard state and code object).
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp, then snapshots a copy of the fx graph
        # magic methods mapping and the current global default device context.
        super().setUp()
        self._fx_magic_methods_snapshot = fx_graph.magic_methods.copy()
        self._saved_default_device_context = getattr(
            torch._GLOBAL_DEVICE_CONTEXT, "device_context", None
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Restores the fx magic methods mapping to the snapshot (clear
        # then update), resets the default device if it changed during the test (to None, or to
        # the saved context's device), then calls the parent tearDown.
        fx_graph.magic_methods.clear()
        fx_graph.magic_methods.update(self._fx_magic_methods_snapshot)

        current_ctx = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
        if current_ctx is not self._saved_default_device_context:
            if self._saved_default_device_context is None:
                torch.set_default_device(None)
            else:
                torch.set_default_device(self._saved_default_device_context.device)

        super().tearDown()

    def _tracefunc(self, frame, event, arg):
        # [FUNCTION DOC] HELPER: Trace function installed with the interpreter settrace. On the
        # first call event only, records a _FrameState (copy of locals, globals, code, builtins)
        # into _frame_state; later events are ignored. Returns nothing, so no per-line tracing.
        if event != "call":
            return

        if self._frame_state is not None:
            return

        self._frame_state = _FrameState(
            f_locals=dict(frame.f_locals),
            f_globals=frame.f_globals,
            f_code=frame.f_code,
            f_builtins=frame.f_builtins,
        )

    def _test_serialization(self, guard_type, fn, *args, **kwargs):
        # [FUNCTION DOC] Runs fn once under a trace to capture its frame, then replays it
        # through a simplified InstructionTranslator to produce guards. Args: guard_type name to
        # keep (guards whose type or derived types match are kept; the filter asserts at least
        # one exists), fn (a module is replaced by its forward), plus positional args and
        # kwargs; a special kwarg _gen_fn is a callable that regenerates kwargs (used for
        # single-use iterators, whose captured locals are overwritten with fresh ones). Builds
        # the reference guard manager, and a second manager with save_guards, pickles state via
        # load_guards_state and loads it with load_guard_manager. Returns the pair (reference
        # manager, loaded manager). Side effects: resets dynamo, sets and clears the trace
        # function and initial_global_state, stores _cached_guards_state and _cached_f_code,
        # clears _frame_state. Raises AssertionError if no frame was captured.
        # kwargs might contain a callable that generates kwargs
        torch._dynamo.reset()
        kwarg_gen_fn = kwargs.get("_gen_fn")
        if kwarg_gen_fn is not None:
            kwargs = kwarg_gen_fn()

        self._frame_state = None
        sys.settrace(self._tracefunc)
        if isinstance(fn, torch.nn.Module):
            fn = fn.forward
        try:
            fn(*args, **kwargs)
        finally:
            sys.settrace(None)

        if self._frame_state is None:
            raise AssertionError("Expected _frame_state to be set after tracing")

        # Set f_locals from regenerated kwargs to handle exhausted input iterators
        # NB: This is super janky and might cause unforeseen problems
        if kwarg_gen_fn is not None:
            kwargs = kwarg_gen_fn()
            for key in self._frame_state.f_locals:
                if key in kwargs and isinstance(kwargs[key], Iterator):
                    self._frame_state.f_locals[key] = kwargs[key]

        def guard_filter_fn(guards):
            # [FUNCTION DOC] HELPER: Guard filter callback: marks each guard as kept if its type
            # or any derived type equals the requested one, asserts at least one guard is kept,
            # and returns the list of booleans.
            ret = [
                g.guard_type == guard_type or guard_type in g.derived_guard_types
                for g in guards
            ]
            self.assertTrue(any(ret))
            return ret

        ref_gm = None
        loaded_gm = None

        def transform(instructions: list, code_options: dict[str, object]):
            # [FUNCTION DOC] HELPER: Code transform callback passed to transform_code_object.
            # Builds an InstructionTranslator over the captured frame state under compile,
            # tracing and metrics contexts, runs it, then constructs the reference guard manager
            # and a save_guards manager, loads the serialized state and loaded guard manager,
            # and stores both in the enclosing variables ref_gm and loaded_gm. Does not produce
            # transformed bytecode of its own.
            """
            The goal is here is not to reimplement dynamo, but just to have a
            simplified version to extract the state from symbolic convert.
            Should not work on all cases, but should work on simple functions
            in this test file.
            """
            nonlocal ref_gm
            nonlocal loaded_gm

            torch._dynamo.convert_frame.initial_global_state = (
                torch._C._dynamo.guards.GlobalStateGuard()
            )
            tracer = InstructionTranslator(
                instructions,
                self._frame_state.f_code,
                self._frame_state.f_locals,
                self._frame_state.f_globals,
                self._frame_state.f_builtins,
                fn.__closure__ or (),
                torch.overrides._get_current_function_mode_stack(),
                code_options,
                torch._dynamo.lookup_backend("eager"),
                one_graph=False,
                export=False,
                export_constraints=None,
                frame_state=None,
                speculation_log=SpeculationLog(),
                exn_vt_stack=ExceptionStack(),
                distributed_state=None,
                package=None,
            )
            with (
                compile_context(
                    CompileContext(CompileId(frame_id=0, frame_compile_id=0))
                ),
                tracing(tracer.output.tracing_context),
                tracer.set_current_tx(),
                get_metrics_context(),
                dynamo_timed(""),
            ):
                tracer.run()

                ref_gm = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                ).guard_manager

                check_fn_manager = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                    save_guards=True,
                )
                guards_state = check_fn_manager.guards_state
                self._cached_guards_state = guards_state
                self._cached_f_code = self._frame_state.f_code
                self.assertIsNotNone(guards_state)
                guards_state = torch._dynamo.package.load_guards_state(guards_state)

                loaded_gm = torch._dynamo.package.load_guard_manager(
                    guards_state,
                    self._frame_state.f_code,
                    self._frame_state.f_globals,
                )

        try:
            transform_code_object(self._frame_state.f_code, transform)
        finally:
            torch._dynamo.convert_frame.initial_global_state = None
            self._frame_state = None

        self.assertIsNotNone(ref_gm)
        self.assertIsNotNone(loaded_gm)
        return ref_gm, loaded_gm

    def _test_check_fn(self, ref, loaded, inputs, expected):
        # [FUNCTION DOC] HELPER: Asserts inputs is a dict, that the reference manager check on
        # it returns the expected boolean, and that the loaded manager gives the same answer.
        self.assertIsInstance(inputs, dict)
        self.assertEqual(ref.check(inputs), expected)
        self.assertEqual(ref.check(inputs), loaded.check(inputs))


@torch._dynamo.config.patch({"strict_precompile": True})
class TestGuardSerialization(TestGuardSerializationBase):
    # [CLASS DOC] Test suite for guard serialization with strict_precompile enabled. Each test
    # uses the base helpers to serialize one guard type (TENSOR_MATCH, TYPE_MATCH, EQUALS_MATCH,
    # CONSTANT_MATCH, iterator, dict, functorch and other guards), then verifies the reloaded
    # guard agrees with the original on matching and non-matching inputs, or that serialization
    # raises PackageError for unsupported guard types.
    def test_function_locals(self):
        # [FUNCTION DOC] Serializes a TENSOR_MATCH guard for a function taking a tensor and a
        # locally defined callable; only checks that serialization succeeds.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one; passed in as the callable
            # argument.
            return x + 1

        def fn(x, g):
            # [FUNCTION DOC] HELPER: Calls the given function on x and adds one.
            return g(x) + 1

        self._test_serialization("TENSOR_MATCH", fn, torch.randn(3), foo)

    def test_tensor_match(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for a float32 size-2 tensor input and checks
        # the loaded guard accepts same dtype and shape, rejects a different size, a different
        # dtype and None.
        def f(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Returns the input plus one.
            return x + 1

        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", f, torch.ones(2, dtype=torch.float32)
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(2, dtype=torch.float32)}, True
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(3, dtype=torch.float32)}, False
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(2, dtype=torch.float64)}, False
        )
        self._test_check_fn(ref, loaded, {"x": None}, False)

    def test_not_present_in_generic_dict(self):
        # [FUNCTION DOC] Serializes NOT_PRESENT_IN_GENERIC_DICT for a module call; guard passes
        # for the module, and fails after an instance attribute forward is assigned on the
        # module.
        class Module(torch.nn.Module):
            # [FUNCTION DOC] HELPER: Local module whose forward adds one.
            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

        m = Module()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the enclosing module on x.
            return m(x)

        ref, loaded = self._test_serialization(
            "NOT_PRESENT_IN_GENERIC_DICT", fn, torch.ones(2, dtype=torch.float32)
        )
        self._test_check_fn(ref, loaded, {"m": m}, True)

        m.forward = types.MethodType(lambda x: x + 2, m)
        self._test_check_fn(ref, loaded, {"m": m}, False)

    def test_hasattr_serialization(self):
        # [FUNCTION DOC] Serializes a HASATTR guard on a module with attribute a; guard passes
        # while present and fails once the attribute is deleted.
        class Module(torch.nn.Module):
            # [FUNCTION DOC] HELPER: Local module that has an integer attribute a equal to 1.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and sets a to 1.
                super().__init__()
                self.a = 1

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Adds attribute a to the input when the module has it,
                # otherwise adds 2.
                if hasattr(self, "a"):
                    return x + self.a
                else:
                    return x + 2

        m = Module()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the enclosing module on x.
            return m(x)

        ref, loaded = self._test_serialization("HASATTR", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"m": m}, True)
        delattr(m, "a")
        self._test_check_fn(ref, loaded, {"m": m}, False)

    def test_type_match(self):
        # [FUNCTION DOC] Checks TYPE_MATCH serialization. A locally defined module class must
        # raise TypeError asking for global scope; a GlobalModule serializes, passes for another
        # GlobalModule and fails for a plain nn.Module. Also walks all leaf guards recursively
        # and asserts type-id guards on m carry a type hint and the class name GlobalModule in
        # their verbose code parts.
        class LocalModule(torch.nn.Module):
            # [FUNCTION DOC] HELPER: Locally scoped module class used to trigger the
            # global-scope error.
            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

        m = LocalModule()

        def fn(m, x):
            # [FUNCTION DOC] HELPER: Calls the given module on x.
            return m(x)

        with self.assertRaisesRegex(
            TypeError, "Please define the class at global scope"
        ):
            self._test_serialization("TYPE_MATCH", fn, m, torch.randn(3))

        m = GlobalModule()
        ref, loaded = self._test_serialization("TYPE_MATCH", fn, m, torch.randn(3))
        self._test_check_fn(ref, loaded, {"m": m}, True)
        self._test_check_fn(ref, loaded, {"m": GlobalModule()}, True)
        self._test_check_fn(ref, loaded, {"m": torch.nn.Module()}, False)

        # Check verbose_code_parts from leaf guards (they include hints)
        def check_leaf_guards(mgr):
            # [FUNCTION DOC] HELPER: Recursively visits a guard manager: for each leaf guard
            # whose verbose code parts mention the type id check on local m, asserts the text
            # includes the type hint marker and GlobalModule; then recurses into child managers.
            for guard in mgr.get_leaf_guards():
                verbose_parts = guard.verbose_code_parts()
                verbose_str = " ".join(verbose_parts)
                if "___check_type_id" in verbose_str and "L['m']" in verbose_str:
                    self.assertIn(
                        "HINT: type",
                        verbose_str,
                        (
                            lambda msg: f"{msg}\n"
                            + (
                                "TYPE_MATCH guard should include 'HINT: type' "
                                f"annotation.\nGuard: {verbose_str}"
                            )
                        ),
                    )
                    self.assertIn(
                        "GlobalModule",
                        verbose_str,
                        (
                            lambda msg: f"{msg}\n"
                            + (
                                "TYPE_MATCH guard should include type name "
                                f"'GlobalModule'.\nGuard: {verbose_str}"
                            )
                        ),
                    )
            for child_mgr in mgr.get_child_managers():
                check_leaf_guards(child_mgr)

        check_leaf_guards(ref.root)

    def test_tensor_subclass_metadata_match(self):
        # [FUNCTION DOC] Checks TENSOR_SUBCLASS_METADATA_MATCH serialization. A locally defined
        # subclass must raise PackageError about global scope. TwoTensor (no extra metadata)
        # round-trips and matches ones_like. Then SubclassWithMeta,
        # SubclassWithCustomMetadataGuard and SubclassWithSubclassInnerTensor are checked with
        # metadata dicts: default comparison fails if foo or bar differ, the custom guard only
        # fails when bar differs.
        class LocalSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Locally defined wrapper tensor subclass (no metadata) used to
            # check that serializing a subclass guard requires a global-scope class.
            @staticmethod
            def __new__(cls, a, outer_size=None, outer_stride=None):
                # [FUNCTION DOC] HELPER: Creates the wrapper subclass instance with geometry and
                # dtype from a.
                if outer_size is None:
                    outer_size = a.size()
                if outer_stride is None:
                    outer_stride = a.stride()

                shape = outer_size
                kwargs = {}
                kwargs["strides"] = outer_stride
                kwargs["storage_offset"] = a.storage_offset()
                kwargs["device"] = a.device
                kwargs["layout"] = a.layout
                kwargs["requires_grad"] = a.requires_grad
                kwargs["dtype"] = a.dtype
                return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

            def __init__(self, a, outer_size=None, outer_stride=None):
                # [FUNCTION DOC] HELPER: Stores the inner tensor as a.
                self.a = a

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch hook: unwraps to inner tensors, runs the op
                # and rewraps tensor results; other results returned as is.
                if kwargs is None:
                    kwargs = {}
                args_a = pytree.tree_map_only(LocalSubclass, lambda x: x.a, args)
                kwargs_a = pytree.tree_map_only(LocalSubclass, lambda x: x.a, kwargs)
                out_a = func(*args_a, **kwargs_a)
                if isinstance(out_a, torch.Tensor):
                    return LocalSubclass(out_a)
                return out_a

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Returns inner tensor name a and no metadata.
                return ["a"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Rebuilds an instance from inner tensors; raises if meta
                # is not None, or if inner is a plain tensor and outer size or stride is
                # missing.
                assert meta is None  # noqa: S101
                a = inner_tensors["a"]
                if type(a) is torch.Tensor:
                    assert outer_size is not None  # noqa: S101
                    assert outer_stride is not None  # noqa: S101
                return LocalSubclass(a, outer_size, outer_stride)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input times two.
            return x * 2

        # === example subclass defined locally (error) ===
        local_sub = LocalSubclass(torch.randn(3))
        with self.assertRaisesRegex(
            PackageError, "Please define the class at global scope"
        ):
            self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, local_sub)

        # === example subclass with None extra metadata ===
        from torch.testing._internal.two_tensor import TwoTensor

        tt = TwoTensor(torch.randn(3), torch.randn(3))
        ref, loaded = self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, tt)
        self._test_check_fn(ref, loaded, {"x": tt}, True)
        self._test_check_fn(ref, loaded, {"x": torch.ones_like(tt)}, True)

        # used below for convenience; returned func accepts some metadata and whether the
        # guard is expected to pass for the given subclass type
        def _get_meta_test_check_fn(ref, loaded, subclass_type):
            # [FUNCTION DOC] HELPER: Given reference and loaded managers and a subclass type,
            # returns a checker function taking metadata and expected result.
            def _f(meta, expected, ref=ref, loaded=loaded, subclass_type=subclass_type):
                # [FUNCTION DOC] HELPER: Builds an instance of the subclass over a random tensor
                # with the given metadata as extra and asserts both guard managers return the
                # expected result for it.
                self._test_check_fn(
                    ref,
                    loaded,
                    {"x": subclass_type(torch.randn(3), extra=meta)},
                    expected,
                )

            return _f

        # === example subclass with extra metadata ===
        extra_meta = {
            "foo": 5,
            "bar": "hello",
        }
        sub = SubclassWithMeta(torch.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(ref, loaded, SubclassWithMeta)
        check_with_meta(dict(extra_meta), True)
        # different "foo"
        check_with_meta({"foo": 6, "bar": "hello"}, False)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

        # === example subclass with custom metadata guard logic ===
        sub = SubclassWithCustomMetadataGuard(torch.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(
            ref, loaded, SubclassWithCustomMetadataGuard
        )
        check_with_meta(dict(extra_meta), True)
        # different "foo"; custom logic says this is okay
        check_with_meta({"foo": 6, "bar": "hello"}, True)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

        # === example subclass with subclass inner tensor ===
        sub = SubclassWithSubclassInnerTensor(torch.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(
            ref, loaded, SubclassWithSubclassInnerTensor
        )
        check_with_meta(dict(extra_meta), True)
        # different "foo"
        check_with_meta({"foo": 6, "bar": "hello"}, False)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    def test_transparent_subclass_tensor_match(self):
        # [FUNCTION DOC] Checks TENSOR_MATCH round-trips for AsyncCollectiveTensor (a
        # transparent wrapper subclass whose empty_like drops the subclass type); the loaded
        # guard must match an equivalent wrapper and also a plain tensor. Skipped if
        # torch.distributed is unavailable.
        # AsyncCollectiveTensor is a transparent traceable wrapper subclass: its
        # __torch_dispatch__ desugars ops to the inner tensor, so
        # torch.empty_like(act) returns a plain Tensor and drops the subclass
        # type. Guard-state serialization must round-trip such an input by
        # unflattening through the recorded pytype rather than type(meta_tensor)
        # (which would be torch.Tensor, with no __tensor_unflatten__).
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        def fn(w):
            # [FUNCTION DOC] HELPER: Returns the sum of the input.
            return w.sum()

        base = torch.randn(3, 4)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", fn, AsyncCollectiveTensor(base)
        )
        self._test_check_fn(ref, loaded, {"w": AsyncCollectiveTensor(base)}, True)
        # The reloaded guard must also match the resolved plain Tensor -- the
        # ACT->Tensor reuse this relaxation enables, and the case that matters
        # for precompile under FSDP+TP.
        self._test_check_fn(ref, loaded, {"w": base}, True)

    def test_equals_match(self):
        # [FUNCTION DOC] Serializes EQUALS_MATCH for membership of a CustomConstantType in a
        # list; passes for equal items (custom equality ignores b), fails for an empty list or a
        # list lacking an equal item.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns zeros if x is in y, else ones.
            # CustomConstantType is registered as a pytree constant so this should
            # result in an EQUALS_MATCH guard.
            if x in y:
                return torch.zeros(3)
            return torch.ones(3)

        x = CustomConstantType(4, 5)
        y = [CustomConstantType(2, 3), CustomConstantType(4, 5)]
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        # custom __eq__ says that CustomConstantType(4, 5) == CustomConstantType(4, 9)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": CustomConstantType(4, 5),
                "y": [CustomConstantType(2, 3), CustomConstantType(4, 9)],
            },
            True,
        )
        self._test_check_fn(ref, loaded, {"x": x, "y": []}, False)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": x,
                "y": [CustomConstantType(2, 3), CustomConstantType(6, 7)],
            },
            False,
        )

    def test_constant_match(self):
        # [FUNCTION DOC] Serializes CONSTANT_MATCH for a bool, a None and an int argument.
        # Loaded guard passes for same constant regardless of tensor shape and fails for
        # different constants (False, non-None values, a different int).
        # === bool constant ===
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds one to x if y is truthy else adds two.
            if y:
                return x + 1
            return x + 2

        x = torch.randn(3)
        y = True

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": True}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(4), "y": True}, True)
        # guard should fail for different y value
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": False}, False)

        # === None constant ===
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds one to x if y is None else adds two.
            if y is None:
                return x + 1
            return x + 2

        x = torch.randn(3)
        y = None

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": None}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(4), "y": None}, True)
        # guard should fail for non-None y value
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": 5}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": True}, False)

        # === int constant ===
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        x = torch.randn(3)
        y = 5

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": 5}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(4), "y": 5}, True)
        # guard should fail for different y value
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": 6}, False)

    def test_class_match(self):
        # [FUNCTION DOC] Asserts that serializing a CLASS_MATCH guard (created by using no_grad
        # as a context manager) raises PackageError saying it cannot be serialized.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies x by two under no_grad and returns the result.
            # usage of this context manager installs a FUNCTION_MATCH guard
            with torch.no_grad():
                y = x * 2
            return y

        x = torch.randn(3)

        # we don't support FUNCTION_MATCH because it adds an ID_MATCH guard, and we don't
        # support that in serialization
        with self.assertRaisesRegex(
            PackageError, "CLASS_MATCH guard cannot be serialized."
        ):
            self._test_serialization("CLASS_MATCH", fn, x)

    def test_closure_match(self):
        # [FUNCTION DOC] Asserts that serializing a CLOSURE_MATCH guard (created by calling a
        # global function) raises PackageError saying it cannot be serialized.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the global function applied to x.
            # usage of this global function installs a CLOSURE_MATCH guard
            return global_func(x)

        x = torch.randn(3)

        # we don't support CLOSURE_MATCH because it adds a FUNCTION_MATCH guard, and we don't
        # support that in serialization
        with self.assertRaisesRegex(
            PackageError, "CLOSURE_MATCH guard cannot be serialized."
        ):
            self._test_serialization("CLOSURE_MATCH", fn, x)

    def test_sequence_length(self):
        # [FUNCTION DOC] Serializes SEQUENCE_LENGTH for a 3-tuple input; passes for any 3-tuple
        # even with different element types, fails for a 4-tuple.
        # tuple input installs a SEQUENCE_LENGTH guard
        def fn(t, x):
            # [FUNCTION DOC] HELPER: Returns the second tuple element plus x.
            return t[1] + x

        t = tuple(torch.randn(3) for _ in range(3))
        x = torch.randn(3)

        ref, loaded = self._test_serialization("SEQUENCE_LENGTH", fn, t, x)
        self._test_check_fn(ref, loaded, {"x": x, "t": t}, True)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": torch.randn(3),
                "t": tuple(torch.randn(3) for _ in range(3)),
            },
            True,
        )
        # different types in tuple of same length shouldn't fail SEQUENCE_LENGTH guard
        # (it should fail the separate TYPE_MATCH guard but that isn't tested here)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "t": (0, 1, 2)}, True)
        # different length tuple
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": torch.randn(3),
                "t": tuple(torch.randn(3) for _ in range(4)),
            },
            False,
        )

    def test_tuple_iterator_len(self):
        # [FUNCTION DOC] Serializes TUPLE_ITERATOR_LEN for a tuple iterator of length 3 (inputs
        # regenerated via a generator callable). Passes for any length-3 iterator regardless of
        # contents or x shape; fails for lengths 2 and 4.
        def fn(t, x):
            # [FUNCTION DOC] HELPER: Multiplies x by two if the iterator yields more than 2
            # items, else adds one.
            if len(list(t)) > 2:
                return x * 2
            return x + 1

        tup = (1, 2, 3)
        x = torch.randn(3)

        # func to generate kwargs; useful for avoiding iterator exhaustion issues
        def _gen_kwargs(tup=tup, x=x):
            # [FUNCTION DOC] HELPER: Returns fresh kwargs with a new iterator over the tuple and
            # x, avoiding exhausted iterators.
            return {"t": iter(tup), "x": x}

        ref, loaded = self._test_serialization(
            "TUPLE_ITERATOR_LEN", fn, _gen_fn=_gen_kwargs
        )

        # same tuple
        self._test_check_fn(ref, loaded, {"t": iter(tup), "x": x}, True)
        self._test_check_fn(ref, loaded, {"t": iter(tup), "x": torch.randn(4)}, True)
        # same length tuple, different contents
        self._test_check_fn(ref, loaded, {"t": iter((3, 2, 1)), "x": x}, True)
        self._test_check_fn(
            ref, loaded, {"t": iter((3, 2, 1)), "x": torch.randn(4)}, True
        )
        # different tuple lengths
        self._test_check_fn(ref, loaded, {"t": iter((1, 2)), "x": x}, False)
        self._test_check_fn(
            ref, loaded, {"t": iter((1, 2)), "x": torch.randn(4)}, False
        )
        self._test_check_fn(ref, loaded, {"t": iter((1, 2, 3, 4)), "x": x}, False)
        self._test_check_fn(
            ref, loaded, {"t": iter((1, 2, 3, 4)), "x": torch.randn(4)}, False
        )

    def test_range_iterator_match(self):
        # [FUNCTION DOC] Serializes RANGE_ITERATOR_MATCH for an iterator over a range with start
        # 2, stop 15, step 3. Passes for same range or a different stop giving equal values;
        # fails for different start, step, or stop changing the produced values.
        def fn(x, r):
            # [FUNCTION DOC] HELPER: Loops over the range iterator, returning x plus the last
            # value (or x if empty).
            y = x
            for val in r:
                y = x + val
            return y

        x = torch.randn(3)

        def _gen_kwargs(x=x):
            # [FUNCTION DOC] HELPER: Returns fresh kwargs with x and a new range iterator.
            return {"x": x, "r": iter(range(2, 15, 3))}

        ref, loaded = self._test_serialization(
            "RANGE_ITERATOR_MATCH", fn, _gen_fn=_gen_kwargs
        )

        # same range
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 15, 3))}, True)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(2, 15, 3))}, True
        )
        # equivalent even with different end
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 16, 3))}, True)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(2, 16, 3))}, True
        )
        # different start
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(1, 15, 3))}, False)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(1, 15, 3))}, False
        )
        # different end resulting in different values
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 18, 3))}, False)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(2, 18, 3))}, False
        )
        # different step
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 15, 4))}, False)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(2, 15, 4))}, False
        )

    def test_count_iterator_match(self):
        # [FUNCTION DOC] Serializes COUNT_ITERATOR_MATCH for an itertools.count starting at 2
        # with step 3; passes for same start and step, fails for different start or step.
        def fn(x, counter):
            # [FUNCTION DOC] HELPER: Adds the next counter value to x.
            return x + next(counter)

        x = torch.randn(3)

        def _gen_kwargs(x=x):
            # [FUNCTION DOC] HELPER: Returns fresh kwargs with x and a new counter starting at 2
            # with step 3.
            return {"x": x, "counter": itertools.count(2, 3)}

        ref, loaded = self._test_serialization(
            "COUNT_ITERATOR_MATCH", fn, _gen_fn=_gen_kwargs
        )

        self._test_check_fn(
            ref, loaded, {"x": x, "counter": itertools.count(2, 3)}, True
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": torch.randn(4), "counter": itertools.count(2, 3)},
            True,
        )
        self._test_check_fn(
            ref, loaded, {"x": x, "counter": itertools.count(5, 3)}, False
        )
        self._test_check_fn(
            ref, loaded, {"x": x, "counter": itertools.count(2, 4)}, False
        )

    def test_supported_nodes_dict_keys_match(self):
        # [FUNCTION DOC] Serializes DICT_KEYS_MATCH for a dict flattened by pytree (supported
        # nodes); passes for same key set, fails for empty dict. Checks the force-keys-match
        # flag survives pickling in the guards state, and that the loaded guard fails after a
        # new pytree node type is registered (supported nodes changed), always deregistering
        # afterwards.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the first pytree leaf of x plus one.
            return pytree.tree_leaves(x)[0] + 1

        ref, loaded = self._test_serialization(
            "DICT_KEYS_MATCH", fn, {"t": torch.randn(3)}
        )
        self._test_check_fn(ref, loaded, {"x": {"t": torch.randn(3)}}, True)
        self._test_check_fn(ref, loaded, {"x": {}}, False)

        # Sticky flag must survive pickling so load keeps keys-match instead of
        # re-promoting SUPPORTED_NODES to DICT_VERSION.
        guards_state = torch._dynamo.package.load_guards_state(
            self._cached_guards_state
        )
        self.assertTrue(
            any(g._force_dict_keys_match for g in guards_state.output_graph.guards)
        )

        # Loaded keys-match guard must observe SUPPORTED_NODES key changes, not
        # only changes to the user input dict.
        class _TmpPytreeNode:
            # [CLASS DOC] HELPER: Temporary container class registered as a pytree node to
            # change the set of supported nodes.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the argument as x.
                self.x = x

        inputs = {"x": {"t": torch.randn(3)}}
        self.assertTrue(loaded.check(inputs))
        try:
            pytree.register_pytree_node(
                _TmpPytreeNode,
                lambda n: ([n.x], None),
                lambda xs, _: _TmpPytreeNode(xs[0]),
            )
            self.assertFalse(loaded.check(inputs))
        finally:
            pytree._deregister_pytree_node(_TmpPytreeNode)

    def test_dict_contains(self):
        # [FUNCTION DOC] Serializes DICT_CONTAINS for a membership check on a dict; passes when
        # key t is present (extra keys fine), fails for empty dict.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x at key t plus one if the dict contains t, else
            # ones.
            if x.__contains__("t"):
                return x["t"] + 1
            else:
                return torch.ones(3)

        ref, loaded = self._test_serialization(
            "DICT_CONTAINS", fn, {"t": torch.randn(3)}
        )

        self._test_check_fn(ref, loaded, {"x": {"t": torch.randn(3)}}, True)
        self._test_check_fn(ref, loaded, {"x": {}}, False)
        self._test_check_fn(
            ref, loaded, {"x": {"t": torch.randn(3), "d": torch.randn(3)}}, True
        )

    def test_bool_match(self):
        # [FUNCTION DOC] Serializes BOOL_MATCH on a true boolean argument; passes for True,
        # fails for False and None.
        def fn(x, b):
            # [FUNCTION DOC] HELPER: Adds one to x if b is truthy else two.
            if b:
                return x + 1
            else:
                return x + 2

        ref, loaded = self._test_serialization("BOOL_MATCH", fn, torch.randn(3), True)

        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": True}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": False}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": None}, False)

    def test_none_match(self):
        # [FUNCTION DOC] Serializes NONE_MATCH on a None argument; passes for None, fails for
        # False and True.
        def fn(x, b):
            # [FUNCTION DOC] HELPER: Adds one to x if b is None else two.
            if b is None:
                return x + 1
            else:
                return x + 2

        ref, loaded = self._test_serialization("NONE_MATCH", fn, torch.randn(3), None)

        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": None}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": False}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": True}, False)

    def test_id_match(self):
        # [FUNCTION DOC] Asserts serializing an ID_MATCH guard (created by using the object id)
        # raises PackageError saying it cannot be serialized.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus the id of x.
            return x + id(x)

        with self.assertRaisesRegex(
            PackageError, "ID_MATCH guard cannot be serialized."
        ):
            self._test_serialization("ID_MATCH", fn, torch.randn(3))

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_id_match_with_config(self):
        # [FUNCTION DOC] With caching_precompile enabled, ID_MATCH and CLASS_MATCH guards do
        # serialize and the loaded guards pass for a fresh tensor input.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus the id of x.
            return x + id(x)

        ref, loaded = self._test_serialization("ID_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies x by two under no_grad and returns the result.
            # usage of this context manager installs a CLASS_MATCH guard
            with torch.no_grad():
                y = x * 2
            return y

        ref, loaded = self._test_serialization("CLASS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)

    def test_dispatch_key_set_match(self):
        # [FUNCTION DOC] Serializes DISPATCH_KEY_SET_MATCH on the dispatch keys of a CPU tensor;
        # passes for same key set, fails for the key set of a meta tensor.
        def fn(x, dks):
            # [FUNCTION DOC] HELPER: Branches on whether the key set has CPU, returning sin of x
            # plus one or x minus one.
            if dks.has("CPU"):
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x = torch.randn(3)
        dks = torch._C._dispatch_keys(x)
        ref, loaded = self._test_serialization("DISPATCH_KEY_SET_MATCH", fn, x, dks)

        self._test_check_fn(ref, loaded, {"x": x, "dks": dks}, True)

        x = torch.randn(3, device="meta")
        dks = torch._C._dispatch_keys(x)
        self._test_check_fn(ref, loaded, {"x": x, "dks": dks}, False)

    def test_dual_level(self):
        # [FUNCTION DOC] Serializes DUAL_LEVEL; passes outside forward-mode AD dual level and
        # fails when checked inside an active dual level context.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one inside a dual level context.
            with torch.autograd.forward_ad.dual_level():
                return x + 1

        x = torch.randn(3)
        ref, loaded = self._test_serialization("DUAL_LEVEL", fn, x)

        self._test_check_fn(ref, loaded, {"x": x}, True)
        with torch.autograd.forward_ad.dual_level():
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_functorch_stack_match(self):
        # [FUNCTION DOC] Serializes FUNCTORCH_STACK_MATCH. With an empty functorch stack guard
        # passes and fails inside vmap or grad nesting. Then simulates compile nested in vmap,
        # grad, jvp plus vmap, functionalize and vmap plus grad by serializing inside a run
        # function and checking guards pass only with exactly the same transform nesting stack
        # and fail otherwise.
        # Test when functorch stack is empty.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns jvp of sin at x with tangent x.
            return torch.func.jvp(torch.sin, (x,), (x,))

        x = torch.randn(3, 4)
        ref, loaded = self._test_serialization("FUNCTORCH_STACK_MATCH", fn, x)

        self._test_check_fn(ref, loaded, {"x": x}, True)
        with torch._functorch.vmap.vmap_increment_nesting(2, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        def fn(x):
            # [FUNCTION DOC] HELPER: Vmaps over a function that vmaps grad of sin.
            def g(x):
                # [FUNCTION DOC] HELPER: Vmaps grad of sin over x.
                return torch.vmap(torch.func.grad(torch.sin))(x)

            return torch.vmap(g)(x)

        x = torch.randn(4, 5)
        ref, loaded = self._test_serialization("FUNCTORCH_STACK_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        with torch._functorch.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Test when there are more than 0 functorch layers.
        # Simulate the case where torch.compile is nested inside eager transforms.

        # Case 1: vmap
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return x.sum()

        ref = loaded = None

        def run(x):
            # [FUNCTION DOC] HELPER: Serializes the current fn for FUNCTORCH_STACK_MATCH (with
            # automatic dynamic shapes disabled so no extra symbolic ints appear), storing the
            # managers in the enclosing ref and loaded, then returns fn applied to x. Used
            # inside functorch transforms.
            nonlocal ref, loaded
            # Turn off automatic dynamic shape to so that functionalization
            # doesn't produce extra SymInt to serialize.
            with torch._dynamo.config.patch(automatic_dynamic_shapes=False):
                ref, loaded = self._test_serialization("FUNCTORCH_STACK_MATCH", fn, x)
            return fn(x)

        torch.vmap(run)(x)

        self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, True)
            with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 2: grad
        x = torch.randn(3, 2)
        ref = loaded = None
        torch.func.grad(run)(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch._functorch.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, True)
            with torch._functorch.eager_transforms.grad_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 3: jvp + vmap
        x = torch.randn(3, 4)
        ref = loaded = None

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns jvp of sin at x with tangent x; this is the binding
            # of fn that the run helper uses for case 3.
            return torch.func.jvp(torch.sin, (x,), (x,))

        torch.func.jvp(torch.vmap(run), (x,), (x,))
        self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.eager_transforms.jvp_increment_nesting():
            with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, True)

        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            with torch._functorch.eager_transforms.jvp_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 4: functionalize
        x = torch.randn(3, 2)
        ref = loaded = None
        torch.func.functionalize(run)(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)

        torch._C._functorch._func_increment_nesting(True)
        try:
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            torch._C._functorch._func_decrement_nesting()

        with torch._functorch.eager_transforms.jvp_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 5: vmap + grad
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return x.sum()

        x = torch.randn(3, 2)
        ref = loaded = None
        torch.vmap(torch.func.grad(run))(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            with torch._functorch.eager_transforms.grad_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, True)

        with torch._functorch.eager_transforms.grad_increment_nesting():
            with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_duplicate_input(self):
        # [FUNCTION DOC] Serializes DUPLICATE_INPUT for the same tensor passed twice; passes
        # when both args are the same object and fails when they differ.
        def fn(x, x_):
            # [FUNCTION DOC] HELPER: Returns the sum of both arguments.
            return x + x_

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization("DUPLICATE_INPUT", fn, x, x)

        self._test_check_fn(ref, loaded, {"x": x, "x_": x}, True)
        self._test_check_fn(ref, loaded, {"x": x, "x_": torch.randn(3, 2)}, False)

    def test_weakref_alive(self):
        # [FUNCTION DOC] Asserts serializing a WEAKREF_ALIVE guard (from an optimizer init group
        # on a linear module with gradients, under grad disabled) raises PackageError saying it
        # cannot be serialized.
        mod = torch.nn.Linear(10, 10, bias=False)
        for p in mod.parameters():
            p.grad = torch.rand_like(p)

        opt = torch.optim.SGD(mod.parameters(), lr=0.1)

        def fn():
            # [FUNCTION DOC] HELPER: Runs the optimizer group init to collect parameters and
            # returns the sum of the first one.
            params = []
            opt._init_group(opt.param_groups[0], params, [], [])
            return params[0].sum()

        with self.assertRaisesRegex(
            PackageError, "WEAKREF_ALIVE guard cannot be serialized"
        ):
            with torch.set_grad_enabled(False):
                self._test_serialization("WEAKREF_ALIVE", fn)

    def test_mapping_keys_check(self):
        # [FUNCTION DOC] Serializes MAPPING_KEYS_CHECK for a mappingproxy with keys a and b;
        # passes for same key order, fails for reordered or fewer keys.
        def fn(mp):
            # [FUNCTION DOC] HELPER: Returns the value at key a plus one.
            return mp["a"] + 1

        mp = types.MappingProxyType({"a": torch.randn(3, 2), "b": torch.randn(3, 2)})
        ref, loaded = self._test_serialization("MAPPING_KEYS_CHECK", fn, mp)
        self._test_check_fn(ref, loaded, {"mp": mp}, True)
        self._test_check_fn(
            ref,
            loaded,
            {
                "mp": types.MappingProxyType(
                    {"b": torch.randn(3, 2), "a": torch.randn(3, 2)}
                )
            },
            False,
        )
        self._test_check_fn(
            ref, loaded, {"mp": types.MappingProxyType({"a": torch.randn(3, 2)})}, False
        )

    def test_dict_keys_match(self):
        # [FUNCTION DOC] Serializes DICT_KEYS_MATCH for a dict iterated over its keys; passes
        # for same keys in order, fails for reordered or fewer keys.
        def fn(x):
            # [FUNCTION DOC] HELPER: Sums the values by iterating the dict keys, starting from
            # one.
            ret = 1
            for k in x:
                ret += x[k]
            return ret

        x = {"a": torch.randn(3, 2), "b": torch.randn(3, 2)}
        ref, loaded = self._test_serialization("DICT_KEYS_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        self._test_check_fn(
            ref,
            loaded,
            {"x": {"b": torch.randn(3, 2), "a": torch.randn(3, 2)}},
            False,
        )
        self._test_check_fn(ref, loaded, {"x": {"a": torch.randn(3, 2)}}, False)

    @torch._dynamo.config.patch("skip_nnmodule_hook_guards", False)
    def test_empty_nn_module_hooks_dict(self):
        # [FUNCTION DOC] With skip_nnmodule_hook_guards disabled, serializes
        # EMPTY_NN_MODULE_HOOKS_DICT; passes initially and fails while a forward hook, forward
        # pre hook or backward hook is registered (removed after each).
        class Module(torch.nn.Module):
            # [FUNCTION DOC] HELPER: Local module whose forward adds one.
            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

        m = Module()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the enclosing module on x.
            return m(x)

        x = torch.ones(2, dtype=torch.float32)
        ref, loaded = self._test_serialization("EMPTY_NN_MODULE_HOOKS_DICT", fn, x)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, True)

        h = m.register_forward_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

        h = m.register_forward_pre_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

        h = m.register_backward_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

    def test_grad_mode(self):
        # [FUNCTION DOC] Serializes GLOBAL_STATE under enabled grad mode; the loaded guard fails
        # when checked under no_grad and passes again under enable_grad.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        x = torch.randn(3, 2)
        with torch.enable_grad():
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
        with torch.no_grad():
            self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch.enable_grad():
            self._test_check_fn(ref, loaded, {"x": x}, True)

    def test_grad_mode_loading(self):
        # [FUNCTION DOC] Serializes GLOBAL_STATE under enable_grad, then under no_grad unpickles
        # the cached guards state by hand and builds a CheckFunctionManager from its output
        # graph and shape code parts. The resulting guard must still fail under no_grad, showing
        # loading is not affected by the current grad mode.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        x = torch.randn(3, 2)
        with torch.enable_grad():
            ref, _ = self._test_serialization("GLOBAL_STATE", fn, x)
        with torch.no_grad():
            # Ensure guards state loading is not affected by the current global grad mode.
            guards_state = pickle.loads(self._cached_guards_state)
            check_fn_manager = CheckFunctionManager(
                self._cached_f_code,
                guards_state.output_graph,
                shape_code_parts=guards_state.shape_code_parts,
            )
            loaded = check_fn_manager.guard_manager
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_deterministic_algorithms(self):
        # [FUNCTION DOC] Serializes GLOBAL_STATE with deterministic algorithms enabled; the
        # guard fails when they are disabled and passes when re-enabled. The original setting is
        # restored at the end.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        deterministic_restore = torch.are_deterministic_algorithms_enabled()
        try:
            x = torch.randn(3, 2)
            torch.use_deterministic_algorithms(True)
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
            torch.use_deterministic_algorithms(False)
            self._test_check_fn(ref, loaded, {"x": x}, False)
            torch.use_deterministic_algorithms(True)
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            torch.use_deterministic_algorithms(deterministic_restore)

    def test_torch_function_state(self):
        # [FUNCTION DOC] Checks TORCH_FUNCTION_STATE: serialized inside a global torch function
        # mode the guard passes inside and fails outside it; GLOBAL_STATE with the mode passes
        # inside and fails with torch function disabled; a locally defined mode must raise
        # PackageError about global scope.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        x = torch.randn(3, 2)

        class LocalTorchFunctionMode(TorchFunctionMode):
            # [CLASS DOC] HELPER: Pass-through torch function mode defined in local scope, used
            # to trigger the global-scope serialization error.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Calls the intercepted function with the given args and
                # kwargs (default empty) and returns the result.
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with GlobalTorchFunctionMode():
            ref, loaded = self._test_serialization("TORCH_FUNCTION_STATE", fn, x)
            self._test_check_fn(ref, loaded, {"x": x}, True)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with GlobalTorchFunctionMode():
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
            self._test_check_fn(ref, loaded, {"x": x}, True)
        with GlobalTorchFunctionMode():
            with torch._C.DisableTorchFunction():
                self._test_check_fn(ref, loaded, {"x": x}, False)
        with self.assertRaisesRegex(
            PackageError,
            "defined in local scope. Please define the class at global scope",
        ):
            with LocalTorchFunctionMode():
                ref, loaded = self._test_serialization("TORCH_FUNCTION_STATE", fn, x)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_training_state(self):
        # [FUNCTION DOC] Serializes GLOBAL_STATE for a function that switches an FSDPParamGroup
        # into the forward training state; guard fails under no_grad and passes under
        # enable_grad. Skipped without a GPU.
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

        param_group = FSDPParamGroup(
            [],  # params: List[nn.Parameter],
            (torch.nn.Linear(1, 1),),  # module: nn.Module,
            None,  # mesh_info: FSDPMeshInfo,
            None,  # post_forward_mesh_info: Optional[FSDPMeshInfo],
            torch.device("cpu"),  # device: torch.device,
            None,  # shard_placement_fn: Optional[Callable],
            None,  # mp_policy: MixedPrecisionPolicy,
            None,  # offload_policy: OffloadPolicy,
        )

        def fn(x):
            # [FUNCTION DOC] HELPER: Enters the forward training state of the param group and
            # returns x plus one if the state is forward, else x minus one.
            with param_group.use_training_state(TrainingState.FORWARD):
                if param_group._training_state == TrainingState.FORWARD:
                    return x + 1
                else:
                    return x - 1

        x = torch.randn(3, 2)

        with torch.enable_grad():
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
        with torch.no_grad():
            self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch.enable_grad():
            self._test_check_fn(ref, loaded, {"x": x}, True)

    def test_default_device(self):
        # [FUNCTION DOC] Serializes DEFAULT_DEVICE with default device cpu; the guard fails with
        # default device meta and passes with cpu again. The original default device is restored
        # at the end.
        device = torch.get_default_device()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        x = torch.randn(3, 2)
        try:
            torch.set_default_device("cpu")
            ref, loaded = self._test_serialization("DEFAULT_DEVICE", fn, x)
            torch.set_default_device("meta")
            self._test_check_fn(ref, loaded, {"x": x}, False)
            torch.set_default_device("cpu")
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            torch.set_default_device(device)

    def test_shape_env(self):
        # [FUNCTION DOC] Serializes SHAPE_ENV for a static tensor, then for tensors with dim 0
        # and dim 1 marked dynamic with range 3 to 10; the loaded guard accepts sizes inside the
        # range and rejects sizes just outside (below 3 and above 10).
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)

        x = torch.randn(3, 2)
        torch._dynamo.mark_dynamic(x, 0, min=3, max=10)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": torch.randn(4, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(10, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(11, 2)}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(2, 2)}, False)

        x = torch.randn(3, 3, 2)
        torch._dynamo.mark_dynamic(x, 1, min=3, max=10)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3, 4, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3, 10, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3, 11, 2)}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3, 2, 2)}, False)

    def test_builtin_match(self):
        # [FUNCTION DOC] Serializes BUILTIN_MATCH created by calling getattr; passes normally
        # and fails after the getattr entry in the builtins dict is replaced by a wrapper, which
        # is restored in a finally block.
        def fn(x):
            # [FUNCTION DOC] HELPER: Reads the shape through getattr and returns x plus its
            # first dimension.
            # usage of getattr() here installs a BUILTIN_MATCH guard
            s = getattr(x, "shape")  # noqa: B009
            return x + s[0]

        x = torch.randn(3)

        ref, loaded = self._test_serialization("BUILTIN_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        getattr_original = getattr

        def getattr_new(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper that forwards all arguments to the original
            # getattr; used to replace the builtin.
            return getattr_original(*args, **kwargs)

        builtins_dict = (
            __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        )
        builtins_dict["getattr"] = getattr_new
        try:
            self._test_check_fn(ref, loaded, {"x": x}, False)
        finally:
            builtins_dict["getattr"] = getattr_original

    def test_skipped_objects(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for a module that holds a code object and a
        # function as attributes (objects that cannot be serialized) along with a parameter;
        # serialization must succeed and the guard pass for the module.
        def foo():
            # [FUNCTION DOC] HELPER: Empty function whose code object is stored on the module.
            pass

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module storing the code object and function of foo plus
            # a random parameter p, whose forward adds one and all parameters.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the attributes code, foo and p.
                super().__init__()
                self.code = foo.__code__
                self.foo = foo
                self.p = torch.nn.Parameter(torch.randn(3, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one plus the sum accumulation over each
                # parameter.
                z = x + 1
                for p in self.parameters():
                    z += p
                return z

        m = Module()
        ref, loaded = self._test_serialization("TENSOR_MATCH", m, torch.randn(3, 2))
        self._test_check_fn(ref, loaded, {"self": m, "x": torch.randn(3, 2)}, True)

    def test_bound_method_input(self):
        # [FUNCTION DOC] Serializes TYPE_MATCH for a bound method of MyClass (whose pickling
        # fails) passed as an input to a module forward; the loaded guard passes with a new
        # bound method.
        class MyModule(torch.nn.Module):
            # [FUNCTION DOC] HELPER: Local module whose forward uses the type id of its first
            # argument.
            def forward(self, foo, x):
                # [FUNCTION DOC] HELPER: Returns x plus the id of the type of foo.
                return x + id(type(foo))

        m = MyModule()
        ref, loaded = self._test_serialization(
            "TYPE_MATCH", m, MyClass().add, torch.randn(3, 2)
        )
        self._test_check_fn(
            ref, loaded, {"self": m, "foo": MyClass().add, "x": torch.randn(3, 2)}, True
        )

    def test_bound_methods_missing(self):
        # [FUNCTION DOC] Serializes TYPE_MATCH for a list holding a local object whose pickling
        # raises; serialization must still work and the loaded guard pass for a fresh list.
        class MyClass:
            # [FUNCTION DOC] HELPER: Local class that cannot be pickled.
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: Pickle hook that always raises NotImplementedError.
                raise NotImplementedError

            def add(self, x):
                # [FUNCTION DOC] HELPER: Returns the argument plus one.
                return x + 1

        def foo(x: torch.Tensor, y: list[MyClass]):
            # [FUNCTION DOC] HELPER: Asserts the list has length one and returns x plus one.
            assert len(y) == 1  # noqa: S101
            return x + 1

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, torch.randn(3, 2), [MyClass()]
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(3, 2), "y": [MyClass()]}, True
        )

    def test_bound_methods_empty(self):
        # [FUNCTION DOC] Serializes TYPE_MATCH for a list containing a bound method of an
        # unpicklable class instance; the loaded guard passes for a fresh equivalent list.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Asserts the first list item is callable and returns x plus
            # one.
            assert callable(y[0])  # noqa: S101
            return x + 1

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, torch.randn(3, 2), [MyClassNotSerializable().add]
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": torch.randn(3, 2), "y": [MyClassNotSerializable().add]},
            True,
        )

    def test_ddp_module(self):
        # [FUNCTION DOC] Compiles a function calling a DistributedDataParallel-wrapped module
        # with a CompilePackage, filtering out unsupported guard types, and asserts exactly one
        # guarded code is recorded and its guards state loads. Sets up a single-process gloo
        # group on a temp file and tears it down; skipped if distributed is unavailable.
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")
        from torch.nn.parallel import DistributedDataParallel as DDP

        tmpfile = tempfile.NamedTemporaryFile()  # noqa: SIM115
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, init_method=f"file://{tmpfile.name}"
        )
        try:
            ddp_model = DDP(GlobalNestedModule())

            def foo(ddp, x):
                # [FUNCTION DOC] HELPER: Calls the DDP model on x.
                return ddp(x)

            unsupported = frozenset(
                torch._dynamo.guards.CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
            )
            x = torch.randn(10)
            package = CompilePackage(foo)
            torch._dynamo.optimize(
                package=package,
                guard_filter_fn=lambda gs: [
                    x.guard_type not in unsupported for x in gs
                ],
            )(foo)(ddp_model, x)
            self.assertEqual(len(package._codes[foo.__code__].guarded_codes), 1)
            torch._dynamo.package.load_guards_state(
                package._codes[foo.__code__].guarded_codes[0].guards_state
            )
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_dict_keys_serialization(self):
        # [FUNCTION DOC] Serializes TYPE_MATCH for a dict keys view input that is iterated; the
        # loaded guard passes for another keys view.
        d = {1: 2, 3: 4}

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Adds each key of y to x in place and returns x.
            for k in y:
                x += k
            return x

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, torch.randn(3, 2), d.keys()
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": torch.randn(3, 2), "y": d.keys()},
            True,
        )

    def test_unserializable_sharded_tensor(self):
        # [FUNCTION DOC] Builds a one-rank ShardedTensor in a single-process gloo group
        # (unserializable object stored in the Inputs holder) and serializes TENSOR_MATCH on the
        # other, serializable field; the loaded guard passes. Skipped if distributed is
        # unavailable; always destroys the group and closes the temp file.
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")

        tmpfile = tempfile.NamedTemporaryFile()  # noqa:SIM115
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, init_method=f"file://{tmpfile.name}"
        )
        try:
            ChunkShardingSpec = dist._shard.sharding_spec.ChunkShardingSpec
            ShardedTensor = dist._shard.sharded_tensor.ShardedTensor
            tensor = torch.arange(2, dtype=torch.int64)
            local_tensor = torch.unsqueeze(torch.cat([tensor, tensor + 2]), 0)

            sharding_dim = 0
            sharding_spec = ChunkShardingSpec(
                dim=sharding_dim,
                placements=[
                    "rank:0/cpu",
                ],
            )
            st = ShardedTensor._init_from_local_tensor(
                local_tensor, sharding_spec, [1, 4]
            )

            def foo(inputs):
                # [FUNCTION DOC] HELPER: Returns the x field of the inputs plus one.
                return inputs.x + 1

            ref, loaded = self._test_serialization(
                "TENSOR_MATCH", foo, Inputs(torch.randn(3, 2), st)
            )
            self._test_check_fn(
                ref, loaded, {"inputs": Inputs(torch.randn(3, 2), st)}, True
            )
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_function_with_wrong_fqn(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for an Inputs holder containing a function
        # whose qualified name does not resolve (global_func_wrong_fqn); serialization works and
        # the loaded guard passes.
        def foo(inputs):
            # [FUNCTION DOC] HELPER: Returns the x field of the inputs plus one.
            return inputs.x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, global_func_wrong_fqn)
        )
        self._test_check_fn(
            ref, loaded, {"inputs": Inputs(x, global_func_wrong_fqn)}, True
        )

    def test_c10d_work(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for an Inputs holder containing an unused c10d
        # Work subclass instance; the guard loads and passes. Skipped if distributed is
        # unavailable.
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")

        Work = dist.distributed_c10d.Work

        class DummyWork(Work):
            # [CLASS DOC] HELPER: Minimal c10d Work subclass with a done flag and a configurable
            # success flag, for a trivial implementation of completion, success, wait and
            # result.
            def __init__(self, should_succeed=True):
                # [FUNCTION DOC] HELPER: Initializes the base Work and sets the done flag false
                # and stores whether the work should succeed.
                super().__init__()
                self._done = False
                self._should_succeed = should_succeed

            def is_completed(self):
                # [FUNCTION DOC] HELPER: Returns whether wait has been called.
                return self._done

            def is_success(self):
                # [FUNCTION DOC] HELPER: Returns the configured success flag.
                return self._should_succeed

            def wait(self, timeout=None):
                # [FUNCTION DOC] HELPER: Marks the work done; raises RuntimeError if configured
                # to fail, otherwise returns the work itself. The timeout argument is ignored.
                self._done = True
                if not self._should_succeed:
                    raise RuntimeError("DummyWork failed")
                return self

            def result(self):
                # [FUNCTION DOC] HELPER: Returns the string dummy_result, or raises RuntimeError
                # if configured to fail.
                if not self._should_succeed:
                    raise RuntimeError("DummyWork failed")
                return "dummy_result"

        def foo(inputs):
            # [FUNCTION DOC] HELPER: Returns the x field of the inputs plus one.
            return inputs.x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, DummyWork())
        )
        self._test_check_fn(ref, loaded, {"inputs": Inputs(x, DummyWork())}, True)

    def test_unused_weakref(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for an Inputs holder containing an unused weak
        # reference; the guard loads and passes.
        def foo(inputs):
            # [FUNCTION DOC] HELPER: Returns the x field of the inputs plus one.
            return inputs.x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, weakref.ref(x))
        )
        self._test_check_fn(ref, loaded, {"inputs": Inputs(x, weakref.ref(x))}, True)

    def test_unused_stream(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for an Inputs holder containing an unused
        # stream; the guard loads and passes. Skipped when no accelerator is available.
        if not torch.accelerator.is_available():
            self.skipTest("Accelerator is not available")

        def foo(inputs):
            # [FUNCTION DOC] HELPER: Returns the x field of the inputs plus one.
            return inputs.x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, torch.Stream())
        )
        self._test_check_fn(ref, loaded, {"inputs": Inputs(x, torch.Stream())}, True)

    def test_unused_process_group(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for an Inputs holder containing the default
        # process group (single-process gloo on a temp file); the guard loads and passes.
        # Skipped if distributed is unavailable; destroys the group and closes the file at the
        # end.
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")

        def foo(inputs):
            # [FUNCTION DOC] HELPER: Returns the x field of the inputs plus one.
            return inputs.x + 1

        tmpfile = tempfile.NamedTemporaryFile()  # noqa: SIM115
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpfile.name}",
            rank=0,
            world_size=1,
        )

        try:
            pg = dist.distributed_c10d._get_default_group()
            x = torch.randn(3, 2)
            ref, loaded = self._test_serialization("TENSOR_MATCH", foo, Inputs(x, pg))
            self._test_check_fn(ref, loaded, {"inputs": Inputs(x, pg)}, True)
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_unserializable_submodule(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for a call of a module whose nested submodule
        # cannot be pickled; serialization succeeds and the guard passes.
        def foo(mod, x):
            # [FUNCTION DOC] HELPER: Calls the module on x.
            return mod(x)

        x = torch.randn(10, 10)
        mod = GlobalNestedModule(ModuleNotSerializable())
        ref, loaded = self._test_serialization("TENSOR_MATCH", foo, mod, x)
        self._test_check_fn(ref, loaded, {"mod": mod, "x": x}, True)

    def test_closure_var_missing(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for a function that is passed a closure
        # capturing a tensor; the guard loads and passes.
        captured = torch.randn(3, 2)

        def bar(x):
            # [FUNCTION DOC] HELPER: Returns x plus the captured tensor.
            return x + captured

        def foo(f, x):
            # [FUNCTION DOC] HELPER: Calls f on x.
            return f(x)

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization("TENSOR_MATCH", foo, bar, x)
        self._test_check_fn(ref, loaded, {"f": bar, "x": x}, True)

    def test_bound_method_patched_forward(self):
        # [FUNCTION DOC] Serializes TYPE_MATCH for the original bound forward of a module whose
        # forward attribute was patched with a plain function; the guard loads and passes.
        def forward(x):
            # [FUNCTION DOC] HELPER: Replacement forward returning x plus one.
            return x + 1

        m = FlatModule()
        m_forward = m.forward
        m.forward = forward

        def foo(f, x):
            # [FUNCTION DOC] HELPER: Asserts f is callable and returns f applied to x.
            assert callable(f)  # noqa: S101
            return f(x)

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization("TYPE_MATCH", foo, m_forward, x)
        self._test_check_fn(ref, loaded, {"f": m_forward, "x": x}, True)

    def test_guard_on_key_order_with_cache(self):
        # [FUNCTION DOC] Serializes DICT_KEYS_MATCH for a module holding a dict; the loaded
        # guard fails for a dict with the same entries in different key order.
        def foo(x, mod):
            # [FUNCTION DOC] HELPER: Multiplies x in place by each value of the module dict and
            # returns x.
            for y in mod.d.values():
                x *= y
            return x

        x = torch.randn(3, 2)
        d = {"a": 1e9, "b": 1e-9}
        ref, loaded = self._test_serialization(
            "DICT_KEYS_MATCH", foo, x, ModWithDict(d)
        )
        self._test_check_fn(
            ref, loaded, {"x": x, "d": ModWithDict({"b": 1e-9, "a": 1e9})}, False
        )

    def test_global_state_guard_filter(self):
        # [FUNCTION DOC] Compiles with the eager backend and the skip-all-guards filter under
        # no_grad, then calls again under enable_grad with recompiles forbidden; must not
        # recompile, showing global state guards are gone.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        x = torch.randn(3, 2)

        with torch.no_grad():
            compiled_fn = torch.compile(
                foo,
                backend="eager",
                options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
            )
            compiled_fn(x)

        # Check global guards are gone.
        with torch.enable_grad(), torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), foo(x))

    def test_torch_function_state_filter(self):
        # [FUNCTION DOC] Compiles with the skip-all-guards filter inside a global torch function
        # mode, then calls outside the mode with recompiles forbidden; must not recompile,
        # showing the torch function state guard is gone.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        x = torch.randn(3, 2)

        with GlobalTorchFunctionMode():
            compiled_fn = torch.compile(
                foo,
                backend="eager",
                options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
            )
            compiled_fn(x)

        # Check global guards are gone.
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), foo(x))

    def test_nested_named_tuple(self):
        # [FUNCTION DOC] Serializes TENSOR_MATCH for a function taking a NamedTuple with ints
        # and a tensor defined in the test; only checks serialization does not raise.
        class NestedTuple(NamedTuple):
            # [CLASS DOC] HELPER: Local named tuple with two ints a and b and a tensor c.
            a: int
            b: int
            c: torch.Tensor

        def fn(x: NestedTuple):
            # [FUNCTION DOC] HELPER: Returns the sum of fields a, b and c.
            return x.a + x.b + x.c

        x = NestedTuple(1, 2, torch.randn(3, 2))

        ref, loaded = self._test_serialization("TENSOR_MATCH", fn, x)

    def test_sdp_backend_serialization(self):
        # [FUNCTION DOC] Serializes EQUALS_MATCH for an SDPBackend enum argument (MATH); the
        # loaded guard passes for MATH and fails for flash, efficient and cudnn backends.
        def fn(x, backend):
            # [FUNCTION DOC] HELPER: Adds 1, 2, 3 or 4 to x depending on whether the backend is
            # MATH, flash, efficient or something else.
            # Use the backend enum in a guard-producing way
            if backend == torch.nn.attention.SDPBackend.MATH:
                return x + 1
            elif backend == torch.nn.attention.SDPBackend.FLASH_ATTENTION:
                return x + 2
            elif backend == torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION:
                return x + 3
            else:
                return x + 4

        x = torch.randn(3, 2)
        backend = torch.nn.attention.SDPBackend.MATH

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x, backend)

        # Test with the same backend
        self._test_check_fn(
            ref, loaded, {"x": x, "backend": torch.nn.attention.SDPBackend.MATH}, True
        )

        # Test with different backends
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": torch.nn.attention.SDPBackend.FLASH_ATTENTION},
            False,
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION},
            False,
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": torch.nn.attention.SDPBackend.CUDNN_ATTENTION},
            False,
        )

    def test_source_serialization(self):
        # [FUNCTION DOC] Checks that two equal LocalSource objects with different cached hash
        # values pickle to identical bytes, so the hash cache is excluded from serialization.
        # Test that "equal" sources with different hashes serialize to the same result
        src1 = LocalSource("x")
        src2 = LocalSource("x")

        # Force different cached hashes to test that serialization excludes _hash
        object.__setattr__(src1, "_hash", 12345)
        object.__setattr__(src2, "_hash", 67890)

        self.assertEqual(pickle.dumps(src1), pickle.dumps(src2))

    def test_source_serialization_init_false_fields(self):
        # [FUNCTION DOC] Checks that a DefaultsSource (with fields not set by init) wrapping a
        # LocalSource survives a pickle round trip and compares equal to the original.
        # Test that source serialization handles fields that are not initialized
        from torch._dynamo.source import DefaultsSource, LocalSource

        base = LocalSource("x")
        source = DefaultsSource(base=base, idx_key=0, is_kw=False)

        # Round-trip through pickle should work even with init=False fields
        restored = pickle.loads(pickle.dumps(source))
        self.assertEqual(source, restored)


class SimpleModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a constant attribute c and a random parameter p; forward
    # adds one and all parameters. Used for the FSDP test.
    def __init__(self, c):
        # [FUNCTION DOC] HELPER: Initializes the module with the given constant c and a random
        # 3x2 Parameter p.
        super().__init__()
        self.c = c
        self.p = torch.nn.Parameter(torch.randn(3, 2))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns x plus one with every parameter added in place.
        z = x + 1
        for p in self.parameters():
            z += p
        return z


if torch.distributed.is_available() and not IS_MACOS:
    from torch.testing._internal.common_fsdp import FSDPTestMultiThread

    @torch._dynamo.config.patch({"strict_precompile": True})
    class TestGuardSerializationFSDP(TestGuardSerializationBase, FSDPTestMultiThread):
        # [CLASS DOC] FSDP variant of the guard serialization tests (defined only when
        # distributed is available and not on macOS), with strict_precompile enabled. Combines
        # the base test case with the multithreaded FSDP test case, so setUp runs both.
        def setUp(self):
            # [FUNCTION DOC] Runs setUp of the serialization base class and then of the FSDP
            # multi-thread test case.
            TestGuardSerializationBase.setUp(self)
            FSDPTestMultiThread.setUp(self)

        @unittest.skipIf(
            TEST_WITH_ASAN or IS_LINUX or TEST_WITH_ROCM,
            "https://github.com/pytorch/pytorch/issues/162793",
        )
        def test_guard_serialization_fsdp_module(self):
            # [FUNCTION DOC] Shards a SimpleModule with fully_shard on a one-device mesh, uses a
            # replicated distributed tensor as input, serializes TENSOR_MATCH and checks the
            # loaded guard passes. Skipped on ASAN, Linux and ROCm (see the linked issue).
            from torch.distributed._tensor import distribute_tensor, Replicate
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            mesh = init_device_mesh(str(torch.get_default_device()), (1,))
            m = SimpleModule(42)
            m = fully_shard(m, mesh=mesh)
            inputs = distribute_tensor(torch.randn(3, 2), mesh, [Replicate()])
            ref, loaded = self._test_serialization("TENSOR_MATCH", m, inputs)
            self._test_check_fn(ref, loaded, {"self": m, "x": inputs}, True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
