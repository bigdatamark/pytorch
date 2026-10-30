# Owner(s): ["module: custom-operators"]
# [MODULE DOC] Tests for opaque objects (v2): user classes derived from CustomClassBase that
# are registered with register_custom_class as symbolic (reference-like) or constant (value-
# like) types and then passed through custom ops under make_fx, AOTAutograd, torch.compile,
# export, inductor, tensor subclasses and pickling. The top of the file defines helper opaque
# classes (Color, OpaqueQueue, RNGState, Counter, NestedCounters, value types such as
# ValueConfig and SizeStore, a hoisted string type), registers them at import time, defines a
# custom op taking a hoisted string, and defines a tensor subclass that carries an opaque
# Counter. The test classes are TestOpaqueObject, which per test defines a private operator
# library, and TestOpaqueGenerator.

import contextlib
import enum
import gc
import random
import re
import unittest
from contextlib import ExitStack
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._custom_class_base import CustomClassBase
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    InductorAndRecordGraphs,
    normalize_gm,
)
from torch._dynamo.utils import counters, counters as dynamo_counters
from torch._functorch import config as functorch_config
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    aot_export_module,
)
from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch._library.effects import EffectType
from torch._library.fake_class_registry import FakeScriptObject, maybe_to_fake_obj
from torch._library.opaque_object import (
    _OPAQUE_TYPES,
    _OPAQUE_TYPES_BY_NAME,
    get_opaque_type_name,
    is_custom_class,
    is_opaque_constant_type,
    MemberType,
    register_custom_class,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._graph_pickler import GraphPickler, Options
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.graph import _illegal_char_regex
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    has_cpp_wrapper_for_device,
    HAS_GPU,
)
from torch.utils._import_utils import import_dill


class Color(CustomClassBase):
    # [CLASS DOC] HELPER: Opaque test type imitating a pybind11 enum. It stores a name and an
    # integer value, exposes them as read-only properties and converts to float via the value.
    # Instances are attached to the class itself (RED, GREEN, BLUE) after the definition, plus a
    # plain float class attribute used to test inlining of literal class attributes. Registered
    # symbolic below.
    """Simulates a pybind11-style enum where class attributes are instances of the class."""

    def __init__(self, name: str, value: int) -> None:
        # [FUNCTION DOC] HELPER: Store the color name and integer value in private attributes.
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        # [FUNCTION DOC] HELPER: Property returning the stored color name string.
        return self._name

    @property
    def value(self) -> int:
        # [FUNCTION DOC] HELPER: Property returning the stored integer value.
        return self._value

    def __float__(self) -> float:
        # [FUNCTION DOC] HELPER: Convert to float by converting the stored integer value.
        return float(self._value)

    @staticmethod
    def RED_STATIC() -> "Color":
        # [FUNCTION DOC] HELPER: Static method returning the class-level RED instance, used to
        # test calling static methods on an opaque class.
        return Color.RED


Color.RED = Color("RED", 1)
Color.GREEN = Color("GREEN", 2)
Color.BLUE = Color("BLUE", 3)
Color.DEFAULT_SCALE = 1.5  # Literal class attribute for testing inlining


class CustomDescriptor:
    # [CLASS DOC] HELPER: Descriptor type whose get hook ignores the instance and always
    # returns the number 42. Used as a class attribute to model a descriptor the opaque-object
    # support rejects.
    def __get__(self, obj, objtype=None):
        # [FUNCTION DOC] HELPER: Descriptor get hook; ignores both arguments and returns the
        # constant 42.
        return 42


# Create a class with an unsupported descriptor
class ColorWithDescriptor(CustomClassBase):
    # [CLASS DOC] HELPER: Opaque test type like Color but with an extra class attribute holding a
    # custom descriptor, which is an unsupported attribute kind. A RED instance is attached after
    # the definition. Used to check that tracing graph-breaks on such attributes.
    def __init__(self, name: str, value: int) -> None:
        # [FUNCTION DOC] HELPER: Store name and integer value in private attributes.
        self._name = name
        self._value = value

    def __float__(self) -> float:
        # [FUNCTION DOC] HELPER: Convert to float via the stored integer value.
        return float(self._value)

    # This is an unsupported descriptor type
    custom_prop = CustomDescriptor()


ColorWithDescriptor.RED = ColorWithDescriptor("RED", 1)


class OpaqueQueue(CustomClassBase):
    # [CLASS DOC] HELPER: Reference-type opaque queue wrapping a Python list of tensors plus a
    # default tensor returned when empty. Keeps push, pop and size call counters for tests to
    # assert on. Registered as symbolic. Mutable state: the list and the counters.
    def __init__(self, queue: list[torch.Tensor], init_tensor_: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Store the backing list and the default tensor and zero the
        # three counters.
        super().__init__()
        self.queue = queue
        self.init_tensor_ = init_tensor_

        # For testing purposes
        self._push_counter = 0
        self._pop_counter = 0
        self._size_counter = 0

    def push(self, tensor: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Count the call and append the tensor to the back of the queue.
        self._push_counter += 1
        self.queue.append(tensor)

    def pop(self) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Count the call; remove and return the oldest tensor, or the
        # default tensor if the queue is empty.
        self._pop_counter += 1
        if len(self.queue) > 0:
            return self.queue.pop(0)
        return self.init_tensor_

    def size(self) -> int:
        # [FUNCTION DOC] HELPER: Count the call and return the number of queued tensors.
        self._size_counter += 1
        return len(self.queue)


class NestedQueue(CustomClassBase):
    # [CLASS DOC] HELPER: Reference-type opaque holder of a single OpaqueQueue. Registered
    # symbolic with the queue attribute used as-is, the getter and pop helper inlined
    # during tracing.
    def __init__(self, q):
        # [FUNCTION DOC] HELPER: Store the wrapped queue.
        self.q = q

    def get_q(self):
        # [FUNCTION DOC] HELPER: Return the wrapped queue.
        return self.q

    def pop_q(self):
        # [FUNCTION DOC] HELPER: Pop from the wrapped queue by calling the queue_pop test
        # operator, so the pop is traceable. Returns a tensor.
        return torch.ops._TestOpaqueObject.queue_pop(self.q)


class RNGState(CustomClassBase):
    # [CLASS DOC] HELPER: Reference-type opaque random-number state. Holds an integer seed, a
    # Python random generator seeded with it, and an unpicklable lambda attribute to test pickling
    # paths. Registered symbolic with guards on the seed.
    def __init__(self, seed):
        # [FUNCTION DOC] HELPER: Store the seed, create a seeded Python random generator, and set
        # a lambda attribute that is deliberately not picklable.
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.dummy = lambda x: x  # test something not pickleable

    def get_seed(self):
        # [FUNCTION DOC] HELPER: Return the stored seed.
        return self.seed

    def noisy_inject(self, x):
        # [FUNCTION DOC] HELPER: Call the noisy_inject test operator on the tensor with this
        # object and return its result tensor.
        return torch.ops._TestOpaqueObject.noisy_inject(x, self)


class OpaqueMultiplier(CustomClassBase):
    # [CLASS DOC] HELPER: Opaque object holding a float multiplier, used by the backward tests.
    # The multiplier is a plain public attribute. Registered symbolic.
    """Opaque object that holds a multiplier value for backward tests."""

    def __init__(self, multiplier: float):
        # [FUNCTION DOC] HELPER: Store the multiplier value.
        self.multiplier = multiplier


class Counter(CustomClassBase):
    # [CLASS DOC] HELPER: Reference-type opaque counter with start and end integers. Equality and
    # hash use both fields. Registered symbolic with guards on the start value. Mutable via the
    # increment method.
    def __init__(self, start, end):
        # [FUNCTION DOC] HELPER: Store start and end.
        self.start = start
        self.end = end

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other object is a Counter with the same start
        # and end.
        return (
            isinstance(other, Counter)
            and self.start == other.start
            and self.end == other.end
        )

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Hash of the pair of start and end.
        return hash((self.start, self.end))

    def get_start_tensor(self):
        # [FUNCTION DOC] HELPER: Return the start value as a zero-dim int64 tensor.
        return torch.scalar_tensor(self.start, dtype=torch.int64)

    @property
    def counter(self):
        # [FUNCTION DOC] HELPER: Property returning the start value as a zero-dim int64 tensor.
        return torch.scalar_tensor(self.start, dtype=torch.int64)

    def increment_counter(self):
        # [FUNCTION DOC] HELPER: Mutate the counter in place by adding one to start.
        # Returns nothing.
        self.start += 1


class NestedCounters(CustomClassBase):
    # [CLASS DOC] HELPER: Reference-type opaque holding either one Counter or a list of Counters
    # in a single attribute. Equality compares that attribute. Registered symbolic with several
    # members exposed for tracing (attribute, getters, indexing).
    def __init__(self, c):
        # [FUNCTION DOC] HELPER: Store the wrapped counter or list of counters.
        self.c = c

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal if the other object is also NestedCounters with an equal
        # wrapped value; otherwise False.
        if not isinstance(other, NestedCounters):
            return False
        return self.c == other.c

    def get_c(self):
        # [FUNCTION DOC] HELPER: Return the wrapped counter or list.
        return self.c

    def get_starts(self):
        # [FUNCTION DOC] HELPER: Return the start values: a list if the wrapped value is a list,
        # otherwise the single start integer.
        if isinstance(self.c, list):
            return [c.start for c in self.c]
        else:
            return self.c.start

    def __getitem__(self, idx):
        # [FUNCTION DOC] HELPER: Index the wrapped list and return a new Counter copying the
        # selected counter's start and end, mimicking device-mesh indexing.
        counter = self.c[idx]
        # Create a new counter to match device mesh's __getitem__
        return Counter(counter.start, counter.end)


class AddModule(CustomClassBase, torch.nn.Module):
    # [CLASS DOC] HELPER: Opaque object that is also an nn.Module whose forward multiplies its two
    # inputs. Registered symbolic; used to test modules as opaque objects.
    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Return the elementwise (or scalar) product of the two inputs.
        return x * y


class ValueConfig(CustomClassBase):
    # [CLASS DOC] HELPER: Constant (value-type) opaque holding a mode string. Equality and hash
    # use the mode, it provides a source-reconstruction representation for FX codegen, and can
    # print its mode.
    def __init__(self, mode: str):
        # [FUNCTION DOC] HELPER: Store the mode string.
        self.mode = mode

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal if other is a ValueConfig with the same mode.
        return isinstance(other, ValueConfig) and self.mode == other.mode

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Hash of the mode string.
        return hash(self.mode)

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: Return a pair: the constructor-style source text for this
        # config with its mode, and a mapping from the class name to the class, for FX graph
        # code generation.
        return f"ValueConfig(mode={self.mode!r})", {"ValueConfig": ValueConfig}

    def print_mode(self):
        # [FUNCTION DOC] HELPER: Print the mode string to stdout (side effect only).
        print(self.mode)


class SizeStore(CustomClassBase):
    # [CLASS DOC] HELPER: Constant (value-type) opaque holding an integer size. Equality and hash
    # use the size; provides an FX source representation and a method returning size plus one.
    def __init__(self, size: int):
        # [FUNCTION DOC] HELPER: Store the integer size.
        self.size = size

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal if other is a SizeStore with the same size.
        return isinstance(other, SizeStore) and self.size == other.size

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Hash of the size.
        return hash(self.size)

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: Return a pair: constructor-style source text with the size, and a
        # mapping from the class name to the class, for FX code generation.
        # Return (repr_string, dict_mapping_name_to_type)
        return f"SizeStore(size={self.size!r})", {"SizeStore": SizeStore}

    def increment_size(self):
        # [FUNCTION DOC] HELPER: Return the size plus one without mutating the object.
        return self.size + 1


class NestedValueSize(CustomClassBase):
    # [CLASS DOC] HELPER: Constant (value-type) opaque nesting a SizeStore and a ValueConfig.
    # Equality compares both; hash combines both hashes; FX representation is built recursively.
    def __init__(self, size: SizeStore, config: ValueConfig):
        # [FUNCTION DOC] HELPER: Store the nested size store and config.
        self.size = size
        self.config = config

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when both nested size and nested config are equal.
        # KNOWN ISSUE: no type check on the other operand, so comparing with an unrelated object
        #     raises an attribute error rather than returning False.
        return self.size == other.size and self.config == other.config

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Combine (bitwise xor) the hashes of the nested size and config.
        return hash(self.size) ^ hash(self.config)

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: Return a pair: source text constructing this object from the
        # nested objects' own source texts, and a merged mapping of class names to classes from
        # this class and both nested objects.
        # Recursively call __fx_repr__ on nested opaque objects
        size_eval, size_globals = self.size.__fx_repr__()
        config_eval, config_globals = self.config.__fx_repr__()

        # Combine repr and globals
        repr_str = f"NestedValueSize(size={size_eval}, config={config_eval})"
        all_globals = (
            {"NestedValueSize": NestedValueSize} | size_globals | config_globals
        )

        return repr_str, all_globals


class HoistedString(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Constant opaque holding a string, registered as hoisted so the value is
    # lifted out of the graph rather than baked in. Equality and hash use the string; FX
    # representation is a constructor-style text with the quoted string.
    def __init__(self, val):
        # [FUNCTION DOC] HELPER: Store the string value.
        self.val = val

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal if the stored strings are equal.
        # KNOWN ISSUE: no type check on the other operand, so a non-HoistedString raises an
        #     attribute error.
        return self.val == other.val

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Hash of the stored string.
        return hash(self.val)

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: Return a pair: constructor-style source text with the string in
        # single quotes, and a mapping from the class name to the class.
        return (f"HoistedString('{self.val}')", {"HoistedString": HoistedString})


register_custom_class(HoistedString, typ="constant", hoist=True)


@torch.library.custom_op("mylib::op_with_string", mutates_args=())
def op_with_string(x: torch.Tensor, s: HoistedString) -> torch.Tensor:
    # [FUNCTION DOC] Custom op (no mutation) taking a tensor and a HoistedString: returns the
    # tensor doubled if the string says double, squared if it says square, and otherwise
    # raises an AssertionError.
    if s.val == "double":
        return x * 2
    elif s.val == "square":
        return x**2
    raise AssertionError("expected double or square")


@op_with_string.register_fake
def _(x, s):
    # [FUNCTION DOC] HELPER: Fake (meta) implementation of the op_with_string op; returns an empty
    # tensor shaped like the input.
    return torch.empty_like(x)


register_custom_class(OpaqueQueue, typ="symbolic")
register_custom_class(
    RNGState,
    typ="symbolic",
    guard_fn=lambda obj: [obj.seed],
    members={
        "seed": MemberType.USE_REAL,
        "get_seed": MemberType.USE_REAL,
        "noisy_inject": MemberType.INLINED,
    },
)
register_custom_class(
    Counter,
    typ="symbolic",
    guard_fn=lambda obj: [obj.start],
    members={"start": MemberType.USE_REAL, "get_start_tensor": MemberType.INLINED},
)
register_custom_class(
    NestedCounters,
    typ="symbolic",
    members={
        "c": MemberType.USE_REAL,
        "get_c": MemberType.USE_REAL,
        "get_starts": MemberType.INLINED,
        "__getitem__": MemberType.INLINED,
    },
)
register_custom_class(
    NestedQueue,
    typ="symbolic",
    members={
        "q": MemberType.USE_REAL,
        "get_q": MemberType.INLINED,
        "pop_q": MemberType.INLINED,
    },
)
register_custom_class(AddModule, typ="symbolic")
register_custom_class(ValueConfig, typ="constant")
register_custom_class(SizeStore, typ="constant")
register_custom_class(NestedValueSize, typ="constant")
register_custom_class(OpaqueMultiplier, typ="symbolic")
register_custom_class(Color, typ="symbolic")
register_custom_class(ColorWithDescriptor, typ="symbolic")


# A tensor subclass (similar to TwoTensor) that also holds an opaque Counter
# object
class TensorWithCounter(torch.Tensor):
    # [CLASS DOC] HELPER: Wrapper tensor subclass (like a two-tensor) holding two inner tensors a
    # and b plus an opaque Counter and an opaque SizeStore. Flattens into a, b and the counter as
    # inner tensor attributes with the size store as metadata. Dispatch unwraps to a, runs the op,
    # and re-wraps tensor outputs with a clone as b and the same counter and size store.
    @staticmethod
    def __new__(cls, a, b, counter, size_store, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] HELPER: Construct the wrapper tensor with the outer size and strides
        # (defaulting to those of a) and a's dtype, device, layout, offset and requires_grad.
        # Raises AssertionError if a and b differ in device or dtype.
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        if not (a.device == b.device and a.dtype == b.dtype):
            raise AssertionError(
                f"device/dtype mismatch: {a.device}/{a.dtype} vs {b.device}/{b.dtype}"
            )
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, outer_size, **kwargs)
        return out

    def __init__(self, a, b, counter, size_store, outer_size=None, outer_stride=None):
        # [FUNCTION DOC] HELPER: Store the inner tensors a and b, the counter and the size store.
        self.a = a
        self.b = b
        self._counter = counter
        self._size_store = size_store

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Return a string showing the inner tensors, counter and
        # size store.
        return f"TensorWithCounter({self.a}, {self.b}, {self._counter}, {self._size_store})"

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Tensor-subclass flatten hook: return the names of the
        # inner attributes (a, b and the counter) and a one-element context tuple holding
        # the size store.
        return ["a", "b", "_counter"], (self._size_store,)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Tensor-subclass unflatten hook: rebuild from the inner mapping
        # and context, passing the outer size and stride through. Returns a new TensorWithCounter.
        a, b = inner_tensors["a"], inner_tensors["b"]
        counter = inner_tensors["_counter"]
        (size_store,) = ctx
        return TensorWithCounter(a, b, counter, size_store, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] HELPER: Dispatch hook. Finds the counter and size store from the first
        # TensorWithCounter among the flattened args, unwraps all such args to their a tensor,
        # runs the op, and wraps every tensor in the output back into a TensorWithCounter (b is a
        # clone). Non-tensors pass through. If no TensorWithCounter is among the args the counter
        # and size store are absent.
        if kwargs is None:
            kwargs = {}

        def unwrap(x):
            # [FUNCTION DOC] HELPER: Return the inner tensor a for a TensorWithCounter, else the
            # value unchanged.
            return x.a if isinstance(x, TensorWithCounter) else x

        def wrap(x, counter, size_store):
            # [FUNCTION DOC] HELPER: Wrap a tensor into a TensorWithCounter using a clone as b and
            # the given counter and size store; non-tensors are returned unchanged.
            return (
                TensorWithCounter(x, x.clone(), counter, size_store)
                if isinstance(x, torch.Tensor)
                else x
            )

        # Get counter from first TensorWithCounter arg
        counter = None
        size_store = None
        for arg in torch.utils._pytree.tree_leaves(args):
            if isinstance(arg, TensorWithCounter):
                counter = arg._counter
                size_store = arg._size_store
                break

        unwrapped_args = torch.utils._pytree.tree_map(unwrap, args)
        unwrapped_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)

        out = func(*unwrapped_args, **unwrapped_kwargs)
        return torch.utils._pytree.tree_map(lambda x: wrap(x, counter, size_store), out)

    @property
    def size_store(self):
        # [FUNCTION DOC] HELPER: Property returning the stored size store.
        return self._size_store

    def get_size_store(self):
        # [FUNCTION DOC] HELPER: Return the stored size store.
        return self._size_store

    def get_counter(self):
        # [FUNCTION DOC] HELPER: Return the stored counter.
        return self._counter


class TestOpaqueObject(TestCase):
    # [CLASS DOC] Main test case for opaque objects. Each test gets a fresh private operator
    # library named _TestOpaqueObject, populated in setUp with ops over the queue, RNG,
    # counter, config, multiplier and color types; tearDown destroys it and unregisters any
    # opaque types registered during the test. Tests cover tracing, AOT export, torch.compile,
    # recompilation and guards, value vs reference types, tensor subclasses with opaque
    # attributes, export and pickling.
    def setUp(self):
        # [FUNCTION DOC] Test fixture setup. Calls the parent setUp first, creates the fragment
        # library, records the set of registered opaque type names, then defines and implements
        # operators: queue_push (effectful, ordered), queue_pop (effectful), queue_size,
        # noisy_inject, increment_counter (mutates its tensor arg), process_with_config,
        # process_nested_config, process_multiple_sizes, apply_color_scale, mul_with_scale (with
        # autograd), multi_rng (with autograd calling a backward op), create_multiplier,
        # create_counter, create_counter2 and counter_start, each with a fake implementation. Side
        # effects: global operator registrations.
        # Must run first: super().setUp() can raise SkipTest (e.g. under
        # PYTORCH_TEST_SKIP_FAST), and unittest skips tearDown when setUp
        # raises. Any registrations before this would leak into the next test.
        super().setUp()

        self.lib = torch.library.Library("_TestOpaqueObject", "FRAGMENT")  # noqa: SCOPED_LIBRARY
        self._opaque_types_before_test = set(_OPAQUE_TYPES_BY_NAME.keys())

        torch.library.define(
            "_TestOpaqueObject::queue_push",
            f"({get_opaque_type_name(OpaqueQueue)} a, Tensor b) -> ()",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::queue_push", "CompositeExplicitAutograd", lib=self.lib
        )
        def push_impl(queue: OpaqueQueue, b: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Real implementation of queue_push; checks the queue type
            # (AssertionError otherwise) and pushes the tensor.
            if not isinstance(queue, OpaqueQueue):
                raise AssertionError(f"expected OpaqueQueue, got {type(queue)}")
            queue.push(b)

        @torch.library.register_fake("_TestOpaqueObject::queue_push", lib=self.lib)
        def push_impl_fake(q: OpaqueQueue, b: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Fake implementation of queue_push; does nothing.
            pass

        torch.library._register_effectful_op(
            "_TestOpaqueObject::queue_push", EffectType.ORDERED
        )

        self.lib.define(
            f"queue_pop({get_opaque_type_name(OpaqueQueue)} a) -> Tensor",
        )

        def pop_impl(queue: OpaqueQueue) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of queue_pop; checks the queue type
            # (AssertionError otherwise) and returns the popped tensor.
            if not isinstance(queue, OpaqueQueue):
                raise AssertionError(f"expected OpaqueQueue, got {type(queue)}")
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")

        def pop_impl_fake(q: OpaqueQueue) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of queue_pop; returns an empty tensor
            # whose length is a new unbacked dynamic size from the fake context.
            # This is not accurate since the queue could have tensors that are
            # not rank 1
            ctx = torch.library.get_ctx()
            u0 = ctx.new_dynamic_size()
            return torch.empty(u0)

        self.lib._register_fake("queue_pop", pop_impl_fake)

        torch.library._register_effectful_op(
            "_TestOpaqueObject::queue_pop", EffectType.ORDERED
        )

        @torch.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
        )
        def size_impl(queue: OpaqueQueue) -> int:
            # [FUNCTION DOC] HELPER: Real implementation of queue_size; checks the queue type
            # (AssertionError otherwise) and returns the queue length.
            if not isinstance(queue, OpaqueQueue):
                raise AssertionError(f"expected OpaqueQueue, got {type(queue)}")
            return queue.size()

        @size_impl.register_fake
        def size_impl_fake(q: OpaqueQueue) -> int:
            # [FUNCTION DOC] HELPER: Fake implementation of queue_size; returns a new dynamic size
            # from the context after asserting it is non-negative.
            ctx = torch._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            torch._check(u0 >= 0)
            return u0

        torch.library.define(
            "_TestOpaqueObject::noisy_inject",
            f"(Tensor x, {get_opaque_type_name(RNGState)} obj) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::noisy_inject", "CompositeExplicitAutograd", lib=self.lib
        )
        def noisy_inject(x: torch.Tensor, rng_state: RNGState) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of noisy_inject: checks the type
            # (AssertionError otherwise), clones the tensor and adds a random number from the
            # object's generator to each element in order. Advances the generator state.
            if not isinstance(rng_state, RNGState):
                raise AssertionError(f"expected RNGState, got {type(rng_state)}")
            out = x.clone()
            for i in range(out.numel()):
                out.view(-1)[i] += rng_state.rng.random()
            return out

        @torch.library.register_fake("_TestOpaqueObject::noisy_inject", lib=self.lib)
        def noisy_inject_fake(x: torch.Tensor, obj: RNGState) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of noisy_inject; checks the object type
            # and a non-negative seed (AssertionError otherwise) and returns an empty tensor like
            # the input.
            if not isinstance(obj, RNGState):
                raise AssertionError(f"expected RNGState, got {type(obj)}")
            if obj.seed < 0:
                raise AssertionError(f"seed must be >= 0, got {obj.seed}")
            return torch.empty_like(x)

        @torch.library.custom_op(
            "_TestOpaqueObject::increment_counter",
            mutates_args=["prev"],
        )
        def increment_counter_impl(c: Counter, prev: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of increment_counter. Checks type
            # (AssertionError otherwise), copies the counter tensor into the mutated argument,
            # increments the counter, and returns the new counter tensor.
            if not isinstance(c, Counter):
                raise AssertionError(f"expected Counter, got {type(c)}")
            prev.copy_(c.counter)
            c.increment_counter()
            return c.counter

        @increment_counter_impl.register_fake
        def increment_counter_fake(c: Counter, prev: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of increment_counter; returns an empty
            # tensor like prev.
            return torch.empty_like(prev)

        torch.library.define(
            "_TestOpaqueObject::process_with_config",
            f"(Tensor x, {get_opaque_type_name(ValueConfig)} config) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::process_with_config",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_with_config_impl(
            x: torch.Tensor, config: ValueConfig
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of process_with_config: checks type
            # (AssertionError otherwise) then returns the square for mode square, the doubled
            # value (sum with itself) for mode double, else a clone.
            if not isinstance(config, ValueConfig):
                raise AssertionError(f"expected ValueConfig, got {type(config)}")
            if config.mode == "square":
                return x * x
            elif config.mode == "double":
                return x + x
            else:
                return x.clone()

        @torch.library.register_fake(
            "_TestOpaqueObject::process_with_config", lib=self.lib
        )
        def process_with_config_fake(
            x: torch.Tensor, config: ValueConfig
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of process_with_config; empty tensor like
            # the input.
            return torch.empty_like(x)

        torch.library.define(
            "_TestOpaqueObject::process_nested_config",
            f"(Tensor x, {get_opaque_type_name(NestedValueSize)} config) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::process_nested_config",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_nested_config_impl(
            x: torch.Tensor, config: NestedValueSize
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of process_nested_config: checks the
            # type (AssertionError otherwise) then uses the nested config mode: square,
            # double, or clone.
            if not isinstance(config, NestedValueSize):
                raise AssertionError(f"expected NestedValueSize, got {type(config)}")
            if config.config.mode == "square":
                return x * x
            elif config.config.mode == "double":
                return x + x
            else:
                return x.clone()

        @torch.library.register_fake(
            "_TestOpaqueObject::process_nested_config", lib=self.lib
        )
        def process_nested_config_fake(
            x: torch.Tensor, config: ValueConfig
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of process_nested_config; empty tensor
            # like the input.
            return torch.empty_like(x)

        torch.library.define(
            "_TestOpaqueObject::process_multiple_sizes",
            f"(Tensor x, {get_opaque_type_name(SizeStore)}[]? sizes) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::process_multiple_sizes",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def process_multiple_sizes_impl(
            x: torch.Tensor, config: list[SizeStore] | None
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of process_multiple_sizes: with no
            # list returns a clone; otherwise returns a clone with each size store's size
            # added in turn.
            if config is None:
                return x.clone()
            else:
                x_res = x.clone()
                for size in config:
                    x_res += size.size
                return x_res

        @torch.library.register_fake(
            "_TestOpaqueObject::process_multiple_sizes", lib=self.lib
        )
        def process_multiple_sizes_fake(
            x: torch.Tensor, config: list[SizeStore] | None
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of process_multiple_sizes; empty tensor
            # like the input.
            return torch.empty_like(x)

        opaque_multiplier_type = get_opaque_type_name(OpaqueMultiplier)
        color_type = get_opaque_type_name(Color)

        torch.library.define(
            "_TestOpaqueObject::apply_color_scale",
            f"({color_type} color, Tensor x) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::apply_color_scale",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def apply_color_scale_impl(color: Color, x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of apply_color_scale: multiplies the
            # tensor by the color's float value.
            return x * float(color)

        @torch.library.register_fake(
            "_TestOpaqueObject::apply_color_scale", lib=self.lib
        )
        def apply_color_scale_fake(color: Color, x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of apply_color_scale; empty tensor like
            # the input.
            return torch.empty_like(x)

        torch.library.define(
            "_TestOpaqueObject::mul_with_scale",
            f"({opaque_multiplier_type} scale_obj, Tensor x) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        torch.library.define(
            "_TestOpaqueObject::get_multiplier_tensor",
            f"({opaque_multiplier_type} scale_obj, Tensor tensor) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::mul_with_scale",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def mul_with_scale_impl(
            scale_obj: OpaqueMultiplier, x: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of mul_with_scale: multiplies the tensor
            # by the object's multiplier.
            return x * scale_obj.multiplier

        @torch.library.register_fake("_TestOpaqueObject::mul_with_scale", lib=self.lib)
        def mul_with_scale_fake(
            scale_obj: OpaqueMultiplier, x: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of mul_with_scale; empty tensor like
            # the input.
            return torch.empty_like(x)

        @torch.library.impl(
            "_TestOpaqueObject::get_multiplier_tensor",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def get_multiplier_tensor_impl(
            scale_obj: OpaqueMultiplier, tensor: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of get_multiplier_tensor: multiplies the
            # tensor by the object's multiplier.
            return tensor * scale_obj.multiplier

        @torch.library.register_fake(
            "_TestOpaqueObject::get_multiplier_tensor", lib=self.lib
        )
        def get_multiplier_tensor_fake(
            scale_obj: OpaqueMultiplier, tensor: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of get_multiplier_tensor; empty tensor
            # like the input.
            return torch.empty_like(tensor)

        def mul_setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup hook for mul_with_scale: saves the opaque
            # scale object (first input) on the context.
            ctx.scale_obj = inputs[0]

        def mul_backward(ctx, grad) -> tuple[torch.Tensor, None]:
            # [FUNCTION DOC] HELPER: Autograd backward for mul_with_scale: returns no
            # gradient for the opaque object and, for the tensor, the incoming gradient
            # scaled via get_multiplier_tensor.
            scale = torch.ops._TestOpaqueObject.get_multiplier_tensor(
                ctx.scale_obj, grad
            )
            return None, scale

        torch.library.register_autograd(
            "_TestOpaqueObject::mul_with_scale",
            mul_backward,
            setup_context=mul_setup_context,
            lib=self.lib,
        )

        rng_state_type = get_opaque_type_name(RNGState)
        # Define forward custom op that takes two opaque objects
        torch.library.define(
            "_TestOpaqueObject::multi_rng",
            f"(Tensor x, {rng_state_type} rng1, {rng_state_type} rng2) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::multi_rng",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def multi_rng_impl(
            x: torch.Tensor, rng1: RNGState, rng2: RNGState
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of multi_rng: draws one random number
            # from each generator and adds both to the tensor. Advances both generators.
            val1 = rng1.rng.random()
            val2 = rng2.rng.random()
            return x + val1 + val2

        @torch.library.register_fake("_TestOpaqueObject::multi_rng", lib=self.lib)
        def multi_rng_fake(
            x: torch.Tensor, rng1: RNGState, rng2: RNGState
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of multi_rng; returns zeros like
            # the input.
            return torch.zeros_like(x)

        # Define backward custom op that also takes two opaque objects
        torch.library.define(
            "_TestOpaqueObject::multi_rng_backward",
            f"(Tensor grad_output, Tensor saved, {rng_state_type} rng1, {rng_state_type} rng2) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::multi_rng_backward",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def multi_rng_backward_impl(
            grad_output: torch.Tensor,
            saved: torch.Tensor,
            rng1: RNGState,
            rng2: RNGState,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of multi_rng_backward: returns the
            # gradient times 1.5, ignoring the saved tensor and the objects.
            return grad_output * 1.5

        @torch.library.register_fake(
            "_TestOpaqueObject::multi_rng_backward", lib=self.lib
        )
        def multi_rng_backward_fake(
            grad_output: torch.Tensor,
            saved: torch.Tensor,
            rng1: RNGState,
            rng2: RNGState,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of multi_rng_backward; returns the
            # gradient itself.
            return grad_output

        def setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup hook for multi_rng: saves the output tensor
            # for backward and stores both opaque RNG objects on the context.
            """Save opaque objects for backward - tests correct positioning."""
            x, rng1, rng2 = inputs
            ctx.save_for_backward(output)
            ctx.rng1 = rng1
            ctx.rng2 = rng2

        def backward(ctx, grad_output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Autograd backward for multi_rng: calls the backward op with
            # the incoming gradient, saved output and both RNG objects; returns the gradient for x
            # and None for the two objects.
            """Backward pass that uses the opaque objects."""
            saved = ctx.saved_tensors[0]
            grad_input = torch.ops._TestOpaqueObject.multi_rng_backward(
                grad_output, saved, ctx.rng1, ctx.rng2
            )
            return grad_input, None, None

        torch.library.register_autograd(
            "_TestOpaqueObject::multi_rng",
            backward,
            setup_context=setup_context,
            lib=self.lib,
        )

        torch.library.define(
            "_TestOpaqueObject::create_multiplier",
            f"(Tensor scale) -> {opaque_multiplier_type}",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::create_multiplier",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def create_multiplier_impl(scale: torch.Tensor) -> OpaqueMultiplier:
            # [FUNCTION DOC] HELPER: Real implementation of create_multiplier: builds an
            # OpaqueMultiplier from the python number inside the scale tensor.
            return OpaqueMultiplier(scale.item())

        @torch.library.register_fake(
            "_TestOpaqueObject::create_multiplier", lib=self.lib
        )
        def create_multiplier_fake(scale: torch.Tensor) -> OpaqueMultiplier:
            # [FUNCTION DOC] HELPER: Fake implementation of create_multiplier: returns a
            # multiplier of 0.0.
            return OpaqueMultiplier(0.0)

        counter_type = get_opaque_type_name(Counter)
        torch.library.define(
            "_TestOpaqueObject::create_counter",
            f"(SymInt start, SymInt end) -> {counter_type}",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::create_counter",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def create_counter_impl(start: int, end: int) -> Counter:
            # [FUNCTION DOC] HELPER: Real implementation of create_counter: builds a Counter from
            # start and end.
            return Counter(start, end)

        @torch.library.register_fake("_TestOpaqueObject::create_counter", lib=self.lib)
        def create_counter_fake(start: int, end: int) -> Counter:
            # [FUNCTION DOC] HELPER: Fake implementation of create_counter: builds a Counter from
            # start and end.
            return Counter(start, end)

        @torch.library.custom_op(
            "_TestOpaqueObject::create_counter2",
            mutates_args=[],
        )
        def create_counter2_impl(start: int, end: int) -> tuple[Counter, Counter]:
            # [FUNCTION DOC] HELPER: Real implementation of create_counter2: returns a pair of
            # Counters, the second with start and end swapped.
            return Counter(start, end), Counter(end, start)

        @create_counter2_impl.register_fake
        def create_counter2_fake(start: int, end: int) -> tuple[Counter, Counter]:
            # [FUNCTION DOC] HELPER: Fake implementation of create_counter2: same pair as the
            # real one.
            return Counter(start, end), Counter(end, start)

        counter_type = get_opaque_type_name(Counter)
        torch.library.define(
            "_TestOpaqueObject::counter_start",
            f"({counter_type} a) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::counter_start",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def counter_start_impl(a: Counter) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of counter_start: returns the start as a
            # zero-dim int64 tensor.
            return torch.scalar_tensor(a.start, dtype=torch.int64)

        @torch.library.register_fake("_TestOpaqueObject::counter_start", lib=self.lib)
        def counter_start_fake(a: Counter) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation of counter_start: returns a zero-dim
            # int64 tensor of 0.
            return torch.scalar_tensor(0, dtype=torch.int64)

    def tearDown(self):
        # [FUNCTION DOC] Test fixture teardown: destroys the operator library, unregisters from
        # the C++ side and from the Python registry every opaque type registered since setUp, then
        # calls the parent tearDown.
        self.lib._destroy()

        # Clean up any opaque types registered during the test
        types_to_remove = (
            set(_OPAQUE_TYPES_BY_NAME.keys()) - self._opaque_types_before_test
        )
        for name in types_to_remove:
            torch._C._unregister_opaque_type(name)
            _OPAQUE_TYPES_BY_NAME.pop(name, None)

        super().tearDown()

    def test_ops(self):
        # [FUNCTION DOC] Eager test: push a tensor on an OpaqueQueue via the custom ops, check
        # size is 1, pop returns the pushed tensor, and size returns to 0.
        queue = OpaqueQueue([], torch.zeros(3))

        torch.ops._TestOpaqueObject.queue_push(queue, torch.ones(3) + 1)
        size = torch.ops._TestOpaqueObject.queue_size(queue)
        self.assertEqual(size, 1)
        popped = torch.ops._TestOpaqueObject.queue_pop(queue)
        self.assertEqual(popped, torch.ones(3) + 1)
        size = torch.ops._TestOpaqueObject.queue_size(queue)
        self.assertEqual(size, 0)

    def test_custom_op_reference_return(self):
        # [FUNCTION DOC] Test compiling a function (inductor, full graph) that creates counters
        # via ops, including a two-output op, and increments each. Checks the returned counters
        # keep the right types and the values show that created reference objects are mutated and
        # returned correctly.
        @torch.compile(backend="inductor", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: creates one counter with one op and two
            # with the pair op, increments each with the input tensor and returns all counters
            # and results.
            c1 = torch.ops._TestOpaqueObject.create_counter(0, 10)
            c2, c3 = torch.ops._TestOpaqueObject.create_counter2(0, 10)
            return (
                c1,
                torch.ops._TestOpaqueObject.increment_counter(c1, x),
                c2,
                torch.ops._TestOpaqueObject.increment_counter(c2, x),
                c3,
                torch.ops._TestOpaqueObject.increment_counter(c3, x),
            )

        x = torch.scalar_tensor(0, dtype=torch.int64)
        c1, r1, c2, r2, c3, r3 = f(x)
        self.assertIsInstance(c1, Counter)
        self.assertIsInstance(c2, Counter)
        self.assertIsInstance(c3, Counter)
        self.assertEqual(c3.start, 11)
        self.assertEqual(c3.end, 0)
        self.assertEqual(x, torch.scalar_tensor(10, dtype=torch.int64))
        self.assertEqual(r1, torch.scalar_tensor(1, dtype=torch.int64))
        self.assertEqual(r2, torch.scalar_tensor(1, dtype=torch.int64))
        self.assertEqual(r3, torch.scalar_tensor(11, dtype=torch.int64))

    def test_custom_op_value_return(self):
        # [FUNCTION DOC] Test that defining a custom op whose return type is a value-type opaque
        # (alone or inside a tuple) raises a ValueError mentioning an unsupported return type.
        with self.assertRaisesRegex(ValueError, "Return has unsupported type"):

            @torch.library.custom_op(
                "_TestOpaqueObject::create_value_config",
                mutates_args=[],
            )
            def create_value_config(mode: str) -> ValueConfig:
                # [FUNCTION DOC] HELPER: Op body whose return type is rejected at definition time;
                # never runs.
                return ValueConfig(mode)

        with self.assertRaisesRegex(ValueError, "Return has unsupported type"):

            @torch.library.custom_op(
                "_TestOpaqueObject::create_value_config_tuple",
                mutates_args=[],
            )
            def create_value_config_tuple(mode: str) -> tuple[ValueConfig, int]:
                # [FUNCTION DOC] HELPER: Op body whose tuple return type is rejected at definition
                # time; never runs.
                return ValueConfig(mode), 0

    def test_fake_script_object_isinstance_per_type(self):
        # [FUNCTION DOC] Test that fake wrappers of different opaque types are instances of their
        # own class and of the fake wrapper class, but not of each other's class.
        queue = OpaqueQueue([], torch.zeros(3))
        rng = RNGState(42)

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode:
            fake_queue = maybe_to_fake_obj(fake_mode, queue)
            fake_rng = maybe_to_fake_obj(fake_mode, rng)

        self.assertIsInstance(fake_queue, OpaqueQueue)
        self.assertIsInstance(fake_rng, RNGState)

        self.assertNotIsInstance(fake_queue, RNGState)
        self.assertNotIsInstance(fake_rng, OpaqueQueue)

        self.assertIsInstance(fake_queue, FakeScriptObject)
        self.assertIsInstance(fake_rng, FakeScriptObject)

    def test_isinstance_opaque_base_covers_all_opaque_types(self):
        # [FUNCTION DOC] Test that an instance check against the opaque base class succeeds for an
        # enum value, a reference-type opaque object and its fake wrapper, and fails for plain int
        # and string.
        # isinstance(x, CustomClassBase) should match all registered opaque types,
        # not just classes that directly subclass CustomClassBase.

        # Value-type opaque (Enum) — registered but doesn't subclass CustomClassBase
        class MyEnum(enum.Enum):
            # [CLASS DOC] HELPER: Local enum with one member, used to check the base-class
            # instance check accepts value-type enums.
            A = 1

        self.assertIsInstance(MyEnum.A, CustomClassBase)

        # Reference-type opaque (subclasses CustomClassBase) — sanity check
        queue = OpaqueQueue([], torch.zeros(3))
        self.assertIsInstance(queue, CustomClassBase)

        # FakeScriptObject wrapping a reference-type opaque
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode:
            fake_queue = maybe_to_fake_obj(fake_mode, queue)
        self.assertIsInstance(fake_queue, FakeScriptObject)
        self.assertIsInstance(fake_queue, CustomClassBase)

        # Non-opaque value should not match
        self.assertNotIsInstance(42, CustomClassBase)
        self.assertNotIsInstance("hello", CustomClassBase)

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx(self, make_fx_tracing_mode):
        # [FUNCTION DOC] Parametrized over fake and symbolic tracing modes. Traces a module that
        # pushes three tensors on a queue, pops twice and reads sizes, with make_fx. Checks the
        # traced graph gives the same result as eager, that the real queue counters were
        # incremented as expected during tracing, and the expected graph text.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward pushes tan, cos and sin of the input to the
            # queue, pops two tensors, reads the size after each pop, and returns the combined
            # result tensor.
            def forward(self, queue, x):
                # [FUNCTION DOC] HELPER: Perform the pushes, pops and size queries via ops and
                # return the sum of the popped tensors combined with the sizes.
                torch.ops._TestOpaqueObject.queue_push(queue, x.tan())
                torch.ops._TestOpaqueObject.queue_push(queue, x.cos())
                torch.ops._TestOpaqueObject.queue_push(queue, x.sin())
                pop1 = torch.ops._TestOpaqueObject.queue_pop(queue)
                size1 = torch.ops._TestOpaqueObject.queue_size(queue)
                pop2 = torch.ops._TestOpaqueObject.queue_pop(queue)
                size2 = torch.ops._TestOpaqueObject.queue_size(queue)
                x_cos = pop1 + size1
                x_sin = pop2 - size2
                return x_sin + x_cos

        q1 = OpaqueQueue([], torch.empty(0).fill_(-1))
        q2 = OpaqueQueue([], torch.empty(0).fill_(-1))

        x = torch.ones(2, 3)
        gm = make_fx(M(), tracing_mode=make_fx_tracing_mode)(q1, x)
        self.assertTrue(torch.allclose(gm(q1, x), M()(q2, x)))
        self.assertEqual(q1._push_counter, 3)
        self.assertEqual(q1._pop_counter, 2)
        self.assertEqual(q1._size_counter, 2)
        self.assertEqual(q1.size(), 1)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    tan = torch.ops.aten.tan.default(arg1_1)
    queue_push = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, tan);  tan = queue_push = None
    cos = torch.ops.aten.cos.default(arg1_1)
    queue_push_1 = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, cos);  cos = queue_push_1 = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_2 = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, sin);  sin = queue_push_2 = None
    queue_pop = torch.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size = torch.ops._TestOpaqueObject.queue_size.default(arg0_1)
    queue_pop_1 = torch.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size_1 = torch.ops._TestOpaqueObject.queue_size.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(queue_pop, queue_size);  queue_pop = queue_size = None
    sub = torch.ops.aten.sub.Tensor(queue_pop_1, queue_size_1);  queue_pop_1 = queue_size_1 = None
    add_1 = torch.ops.aten.add.Tensor(sub, add);  sub = add = None
    return add_1
    """,
        )

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx_value_type(self, make_fx_tracing_mode):
        # [FUNCTION DOC] Parametrized over fake and symbolic tracing modes. Traces the config op
        # with a value-type config through make_fx and checks the result matches eager and that
        # the generated code embeds the config as a constructor-style literal.
        def f(x, cfg):
            # [FUNCTION DOC] HELPER: Apply the process_with_config op to the tensor and config.
            return torch.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = torch.randn(3, 3)
        cfg = ValueConfig("square")
        gm = make_fx(f, tracing_mode=make_fx_tracing_mode)(x, cfg)
        self.assertEqual(gm(x, cfg), f(x, cfg))

        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, x_1, cfg_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(x_1, ValueConfig(mode='square'));  x_1 = None
    return process_with_config
    """,
        )

    def test_subclass_opaque_output_reuses_input_proxy(self):
        # [FUNCTION DOC] Regression test: compile with AOT eager a function on a subclass
        # whose dispatch re-wraps outputs with the real opaque object instead of the fake
        # proxy. Checks the forward graph reuses the input placeholder and has no constant
        # attribute fetch nodes.
        # Regression test: when a tensor subclass's __torch_dispatch__ wraps
        # the output with the real CustomClassBase (not the FakeScriptObject proxy),
        # the AOTAutograd forward graph should still reference the opaque via
        # its input placeholder — not create a duplicate get_attr constant.
        #
        # This mirrors DTensor where C++ dispatch creates new DTensors storing
        # the real DeviceMesh, so output flattening calls maybe_to_fake_obj
        # and mints a fresh FakeScriptObject for the same underlying object.

        class TensorWithRealCounter(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass with one inner data tensor and one
            # opaque counter as inner attributes, whose dispatch re-wraps outputs with the real
            # (unwrapped from fake) counter.
            @staticmethod
            def __new__(cls, data, counter):
                # [FUNCTION DOC] HELPER: Create the wrapper tensor with the metadata of the
                # data tensor.
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.size(),
                    strides=data.stride(),
                    storage_offset=data.storage_offset(),
                    device=data.device,
                    dtype=data.dtype,
                    layout=data.layout,
                    requires_grad=data.requires_grad,
                )

            def __init__(self, data, counter):
                # [FUNCTION DOC] HELPER: Store the data tensor and the counter.
                self._data = data
                self._counter = counter

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return a fixed placeholder string.
                return "TensorWithRealCounter(...)"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook: names of the data and counter inner
                # attributes and empty context.
                return ["_data", "_counter"], ()

            @staticmethod
            def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Unflatten hook: rebuild from the inner data and counter;
                # sizes are ignored.
                return TensorWithRealCounter(
                    inner_tensors["_data"], inner_tensors["_counter"]
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch hook: finds the counter in the args, unwraps
                # subclass args to the data tensor, runs the op, converts a fake script object
                # counter to its real object, and wraps tensor outputs with that real counter.
                if kwargs is None:
                    kwargs = {}

                counter = None
                for arg in torch.utils._pytree.tree_leaves(args):
                    if isinstance(arg, TensorWithRealCounter):
                        counter = arg._counter
                        break

                def unwrap(x):
                    # [FUNCTION DOC] HELPER: Return the inner data tensor for subclass instances,
                    # else the value as is.
                    return x._data if isinstance(x, TensorWithRealCounter) else x

                out = func(
                    *torch.utils._pytree.tree_map(unwrap, args),
                    **torch.utils._pytree.tree_map(unwrap, kwargs),
                )

                # Unwrap FakeScriptObject to real CustomClassBase, simulating what
                # happens in DTensor's C++ dispatch path.
                real_counter = counter
                if isinstance(counter, FakeScriptObject):
                    real_counter = counter.real_obj

                return torch.utils._pytree.tree_map(
                    lambda x: TensorWithRealCounter(x, real_counter)
                    if isinstance(x, torch.Tensor)
                    else x,
                    out,
                )

        counter = Counter(start=0, end=10)
        x = TensorWithRealCounter(torch.randn(4), counter)

        backend = AotEagerAndRecordGraphs()
        torch.compile(lambda x: x + 1, fullgraph=True, backend=backend)(x)

        fw = backend.fw_graphs[0]
        get_attr_nodes = [n for n in fw.graph.nodes if n.op == "get_attr"]
        self.assertEqual(
            get_attr_nodes,
            [],
            "Opaque output should reuse the input placeholder, not create a get_attr constant",
        )

    def test_guard_pickle_subclass_with_opaque_inner_attr(self):
        # [FUNCTION DOC] Regression test: the guard-state pickler's reducer override for a tensor
        # subclass with opaque inner attributes must work. Builds a TensorWithCounter, runs the
        # reducer and checks that rebuilding from the result gives a tensor.
        # Regression test: the guard state pickler serializes tensor subclasses
        # by iterating over __tensor_flatten__ inner attrs.  Opaque inner attrs
        # (e.g. Counter) must be handled correctly — they are pickled by the
        # normal pickle machinery alongside tensor inner attrs.
        from torch._dynamo.guards import GuardsStatePickler

        a = torch.randn(4)
        b = torch.randn(4)
        counter = Counter(start=0, end=10)
        size = SizeStore(4)
        x = TensorWithCounter(a, b, counter, size)

        import io

        buf = io.BytesIO()
        pickler = GuardsStatePickler({id(x): x}, {}, {}, buf)
        func, args = pickler.reducer_override(x)
        obj = func(*args)
        self.assertIsInstance(obj, torch.Tensor)

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_bad_fake(self, make_fx_tracing_mode):
        # [FUNCTION DOC] Parametrized over fake and symbolic tracing modes. Registers fakes that
        # misbehave (reading an attribute of, then assigning an attribute on, the fake opaque
        # object) and checks make_fx raises an AttributeError with messages about getattr and
        # setattr respectively.
        torch.library.define(
            "_TestOpaqueObject::bad_fake",
            f"(Tensor x, {get_opaque_type_name(RNGState)} obj) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        def f(q, x):
            # [FUNCTION DOC] HELPER: Call the bad_fake op for its side effect then return the
            # cosine of x.
            torch.ops._TestOpaqueObject.bad_fake(x, q)
            return x.cos()

        def bad_fake1(x, rng_state) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Bad fake that asserts it got a fake script object and
            # then reads an attribute of it, which is not allowed; the final line is never
            # reached normally.
            self.assertTrue(isinstance(rng_state, FakeScriptObject))
            out = x.clone()
            for i in range(out.numel()):
                out.view(-1)[i] += rng_state.rng.random()  # bad: accessing attributes
            return out

        torch.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake1, lib=self.lib
        )

        with self.assertRaisesRegex(
            AttributeError,
            "Tried to call __getattr__ with attr",
        ):
            make_fx(f, tracing_mode=make_fx_tracing_mode)(RNGState(0), torch.ones(3))

        def bad_fake2(x, rng_state) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Bad fake that assigns an attribute on the fake opaque object
            # (not allowed), then returns an empty tensor like the input.
            rng_state.rng = "foo"
            return torch.empty_like(x)

        torch.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake2, lib=self.lib
        )

        with self.assertRaisesRegex(
            AttributeError,
            "Tried to call __setattr__ with attr",
        ):
            make_fx(f, tracing_mode=make_fx_tracing_mode)(RNGState(0), torch.ones(3))

    def test_aot_export(self):
        # [FUNCTION DOC] Test aot_export_module on a model applying noisy_inject twice. First
        # checks the graph without effect tokens by default; then registers the op as ordered-
        # effectful and checks the graph threads effect tokens; finally unregisters the effect in
        # a finally block.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying noisy_inject, squaring, noisy_inject, doubling;
            # returns a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Plain constructor calling the parent constructor.
                super().__init__()

            def forward(self, rng_state, x):
                # [FUNCTION DOC] HELPER: Apply the noisy_inject op, square, apply it again,
                # double; return a one-element tuple.
                x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
                x = x * x
                x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
                x = x + x
                return (x,)

        mod = Model()
        rng = RNGState(0)
        x = torch.ones(2, 3)

        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fake_rng = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, rng)
        fake_x = fake_mode.from_tensor(x)
        gm = aot_export_module(mod, (fake_rng, fake_x), trace_joint=False)[0]

        # By default we don't register ops containing PyObjs as being effectful
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    noisy_inject = torch.ops._TestOpaqueObject.noisy_inject.default(arg1_1, arg0_1);  arg1_1 = None
    mul = torch.ops.aten.mul.Tensor(noisy_inject, noisy_inject);  noisy_inject = None
    noisy_inject_1 = torch.ops._TestOpaqueObject.noisy_inject.default(mul, arg0_1);  mul = arg0_1 = None
    add = torch.ops.aten.add.Tensor(noisy_inject_1, noisy_inject_1);  noisy_inject_1 = None
    return (add,)""",
        )

        torch.library._register_effectful_op(
            "_TestOpaqueObject::noisy_inject", EffectType.ORDERED
        )
        try:
            gm = aot_export_module(mod, (rng, fake_x), trace_joint=False)[0]
            # inputs: token, rng, x
            # return: token, res
            self.assertExpectedInline(
                gm.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TestOpaqueObject.noisy_inject.default, arg2_1, arg1_1);  arg0_1 = arg2_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    mul = torch.ops.aten.mul.Tensor(getitem_1, getitem_1);  getitem_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TestOpaqueObject.noisy_inject.default, mul, arg1_1);  getitem = mul = arg1_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(getitem_3, getitem_3);  getitem_3 = None
    return (getitem_2, add)""",
            )
        finally:
            torch.library._register_effectful_op(
                "_TestOpaqueObject::noisy_inject", None
            )

    def test_install_free_opaque_object(self):
        # [FUNCTION DOC] Test that dynamo graph capture for export of a function closing over an
        # opaque object installs it as a graph attribute (not an input) and checks the generated
        # code text.
        rng = RNGState(0)
        x = torch.ones(2, 3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply noisy_inject to x using the captured RNG state.
            return torch.ops._TestOpaqueObject.noisy_inject(x, rng)

        gm = _dynamo_graph_capture_for_export(fn)(x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    arg_0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    l_flat_args_0_ = arg_0
    l__self____export_root___closure___0_cell_contents = self.L__self____export_root___closure___0_cell_contents
    res = torch.ops._TestOpaqueObject.noisy_inject(l_flat_args_0_, l__self____export_root___closure___0_cell_contents);  l_flat_args_0_ = l__self____export_root___closure___0_cell_contents = None
    return pytree.tree_unflatten((res,), self._out_spec)""",
        )

    def test_compile1(self):
        # [FUNCTION DOC] Compile with inductor then with AOT eager a function applying
        # noisy_inject twice. Checks inductor output differs from the noise-free value and the
        # dynamo and AOT graph texts, including the sanitized opaque type name in the signature.
        def foo(rng_state, x):
            # [FUNCTION DOC] HELPER: Raise AssertionError if the first argument is not an
            # RNGState; apply noisy_inject, square, noisy_inject, double.
            if not isinstance(rng_state, RNGState):
                raise AssertionError(f"expected RNGState, got {type(rng_state)}")
            x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
            x = x * x
            x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
            x = x + x
            return x

        rng = RNGState(0)
        x = torch.ones(2, 3)

        res = torch.compile(foo, fullgraph=True, backend="inductor")(rng, x)
        self.assertFalse(torch.allclose(res, x * x + x))

        backend = AotEagerAndRecordGraphs()
        torch.compile(foo, fullgraph=True, backend=backend)(rng, x)

        # This is done in torch.fx's graph in _namespace.create_name() where it
        # sanitizes the name
        fx_class = _illegal_char_regex.sub("_", get_opaque_type_name(RNGState))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            f"""\
def forward(self, L_rng_state_ : {fx_class}, L_x_ : torch.Tensor):
    l_rng_state_ = L_rng_state_
    l_x_ = L_x_
    noisy_inject = torch.ops._TestOpaqueObject.noisy_inject(l_x_, l_rng_state_);  l_x_ = None
    mul = noisy_inject * noisy_inject;  noisy_inject = None
    noisy_inject_1 = torch.ops._TestOpaqueObject.noisy_inject(mul, l_rng_state_);  mul = l_rng_state_ = None
    add = noisy_inject_1 + noisy_inject_1;  noisy_inject_1 = None
    return (add,)""",
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    noisy_inject = torch.ops._TestOpaqueObject.noisy_inject.default(arg1_1, arg0_1);  arg1_1 = None
    mul = torch.ops.aten.mul.Tensor(noisy_inject, noisy_inject);  noisy_inject = None
    noisy_inject_1 = torch.ops._TestOpaqueObject.noisy_inject.default(mul, arg0_1);  mul = arg0_1 = None
    add = torch.ops.aten.add.Tensor(noisy_inject_1, noisy_inject_1);  noisy_inject_1 = None
    return (add,)""",
        )

    def test_compile_inline_methods(self):
        # [FUNCTION DOC] Test inlining of members: compile a function using the real getters
        # (seed) and an inlined method of the RNG object. Checks the AOT graph uses constants for
        # seed values and calls the op for the method, and that inductor output differs from the
        # noise-free value.
        def foo(rng_state, x):
            # [FUNCTION DOC] HELPER: Read the seed via method and attribute, apply noisy_inject,
            # scale by a python number derived from the seeds, call the inlined method, double.
            seed1 = rng_state.get_seed()
            seed2 = rng_state.seed
            x = torch.ops._TestOpaqueObject.noisy_inject(x, rng_state)
            x = x * (seed1 + seed2 + 1)
            x = rng_state.noisy_inject(x)
            x = x + x
            return x

        rng = RNGState(0)
        x = torch.ones(2, 3)

        backend = AotEagerAndRecordGraphs()
        torch.compile(foo, fullgraph=True, backend=backend)(rng, x)

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    noisy_inject = torch.ops._TestOpaqueObject.noisy_inject.default(arg1_1, arg0_1);  arg1_1 = None
    mul = torch.ops.aten.mul.Tensor(noisy_inject, 1);  noisy_inject = None
    noisy_inject_1 = torch.ops._TestOpaqueObject.noisy_inject.default(mul, arg0_1);  mul = arg0_1 = None
    add = torch.ops.aten.add.Tensor(noisy_inject_1, noisy_inject_1);  noisy_inject_1 = None
    return (add,)""",
        )

        res = torch.compile(foo, fullgraph=True, backend="inductor")(rng, x)
        self.assertFalse(torch.allclose(res, x * x + x))

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/184597")
    def test_reference_type_recompile(self):
        # [FUNCTION DOC] Skipped on Linux (linked issue). Checks reference-type recompilation:
        # counters with the same start do not recompile when only end differs, and a different
        # start recompiles.
        cnt = CompileCounter()

        def foo(counter, x):
            # [FUNCTION DOC] HELPER: Increment the counter with an op and multiply x by
            # the result.
            z = torch.ops._TestOpaqueObject.increment_counter(counter, x)
            x = x * z
            return x

        x = torch.ones(2, 3)

        opt_f = torch.compile(foo, backend=cnt, fullgraph=True)
        opt_f(Counter(1, 5), x)
        self.assertEqual(cnt.frame_count, 1)

        opt_f(Counter(1, 6), x)  # we only guard on the first number
        self.assertEqual(cnt.frame_count, 1)

        opt_f(Counter(2, 5), x)  # recompile!
        self.assertEqual(cnt.frame_count, 2)

    def test_nested_reference_recompile(self):
        # [FUNCTION DOC] Same as the recompile test but through a nested counters object: only the
        # nested start value is guarded, so a different end does not recompile and a different
        # start does.
        def foo(nested_counter, x):
            # [FUNCTION DOC] HELPER: Return the nested counter's start added to x.
            c1 = nested_counter.c
            return c1.start + x

        cnt = CompileCounter()
        x = torch.ones(2, 3)
        inp = (NestedCounters(Counter(1, 5)), x)
        opt_f = torch.compile(foo, backend=cnt, fullgraph=True)
        res = opt_f(*inp)
        self.assertEqual(res, foo(*inp))
        self.assertEqual(cnt.frame_count, 1)

        inp = (NestedCounters(Counter(1, 6)), x)
        res = opt_f(*inp)
        self.assertEqual(res, foo(*inp))
        self.assertEqual(cnt.frame_count, 1)  # we only guard on the first number

        inp = (NestedCounters(Counter(2, 5)), x)
        res = opt_f(*inp)
        self.assertEqual(res, foo(*inp))
        self.assertEqual(cnt.frame_count, 2)  # recompile!

    def test_nested_reference_list_trace(self):
        # [FUNCTION DOC] Checks the traced graph for a nested list of counters: each
        # counter becomes a graph input, ops increment each in turn, and starts are folded
        # in as constants.
        def foo(nested_counter, x):
            # [FUNCTION DOC] HELPER: Increment each nested counter with the op then add each
            # start value.
            for c in nested_counter.c:
                x = torch.ops._TestOpaqueObject.increment_counter(c, x)
            for start in nested_counter.get_starts():
                x = x + start
            return x

        backend = AotEagerAndRecordGraphs()
        inp = (NestedCounters([Counter(1, 5), Counter(2, 5)]), torch.ones(2, 3))
        torch.compile(foo, backend=backend, fullgraph=True)(*inp)

        fx_class = _illegal_char_regex.sub("_", get_opaque_type_name(Counter))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            f"""\
def forward(self, L_nested_counter_c_0_ : {fx_class}, L_nested_counter_c_1_ : {fx_class}, L_x_ : torch.Tensor):
    l_nested_counter_c_0_ = L_nested_counter_c_0_
    l_nested_counter_c_1_ = L_nested_counter_c_1_
    l_x_ = L_x_
    increment_counter = torch.ops._TestOpaqueObject.increment_counter(l_nested_counter_c_0_, l_x_);  l_nested_counter_c_0_ = l_x_ = None
    increment_counter_1 = torch.ops._TestOpaqueObject.increment_counter(l_nested_counter_c_1_, increment_counter);  l_nested_counter_c_1_ = increment_counter = None
    add = increment_counter_1 + 1;  increment_counter_1 = None
    add_1 = add + 2;  add = None
    return (add_1,)""",
        )

    def test_nested_reference_trace(self):
        # [FUNCTION DOC] Compile with a nested queue: the queue is lifted as one graph input,
        # pushes and pops are traced, and the AOT forward graph threads effect tokens. Checks
        # equality with eager and the graph texts.
        def foo(nested_queue, x):
            # [FUNCTION DOC] HELPER: Push tan and cos of x via the attribute and the getter, pop
            # twice via the inlined method, return the sum.
            q1 = nested_queue.q
            torch.ops._TestOpaqueObject.queue_push(q1, x.tan())
            q2 = nested_queue.get_q()
            torch.ops._TestOpaqueObject.queue_push(q2, x.cos())
            pop1 = nested_queue.pop_q()
            pop2 = nested_queue.pop_q()
            return pop1 + pop2

        inp = (
            NestedQueue(OpaqueQueue([], torch.empty(0).fill_(-1))),
            torch.randn(2, 3),
        )
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(foo, fullgraph=True, backend=backend)(*inp)
        self.assertEqual(res, foo(*inp))

        fx_class = _illegal_char_regex.sub("_", get_opaque_type_name(OpaqueQueue))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            f"""\
def forward(self, L_nested_queue_q : {fx_class}, L_x_ : torch.Tensor):
    l_nested_queue_q = L_nested_queue_q
    l_x_ = L_x_
    tan = l_x_.tan()
    queue_push = torch.ops._TestOpaqueObject.queue_push(l_nested_queue_q, tan);  tan = queue_push = None
    cos = l_x_.cos();  l_x_ = None
    queue_push_1 = torch.ops._TestOpaqueObject.queue_push(l_nested_queue_q, cos);  cos = queue_push_1 = None
    queue_pop = torch.ops._TestOpaqueObject.queue_pop(l_nested_queue_q)
    sym_size_int = torch.ops.aten.sym_size.int(queue_pop, 0)
    ge = sym_size_int >= 0
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
    queue_pop_1 = torch.ops._TestOpaqueObject.queue_pop(l_nested_queue_q);  l_nested_queue_q = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(queue_pop_1, 0)
    ge_1 = sym_size_int_1 >= 0
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
    eq = sym_size_int == sym_size_int_1;  sym_size_int = sym_size_int_1 = None
    _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u0, u1) on node 'eq'");  eq = _assert_scalar_default_2 = None
    add = queue_pop + queue_pop_1;  queue_pop = queue_pop_1 = None
    return (add,)""",
        )

        # inputs: (token, nested_queue.q, x)
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    tan = torch.ops.aten.tan.default(arg2_1)
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TestOpaqueObject.queue_push.default, arg1_1, tan);  arg0_1 = tan = None
    getitem = with_effects[0];  with_effects = None
    cos = torch.ops.aten.cos.default(arg2_1);  arg2_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TestOpaqueObject.queue_push.default, arg1_1, cos);  getitem = cos = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    with_effects_2 = torch.ops.higher_order.with_effects(getitem_2, torch.ops._TestOpaqueObject.queue_pop.default, arg1_1);  getitem_2 = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    sym_size_int = torch.ops.aten.sym_size.int(getitem_5, 0)
    ge = sym_size_int >= 0
    _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._TestOpaqueObject.queue_pop.default, arg1_1);  getitem_4 = arg1_1 = None
    getitem_6 = with_effects_3[0]
    getitem_7 = with_effects_3[1];  with_effects_3 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(getitem_7, 0)
    ge_1 = sym_size_int_1 >= 0
    _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
    eq_2 = sym_size_int == sym_size_int_1;  sym_size_int = sym_size_int_1 = None
    _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq_2, "Runtime assertion failed for expression Eq(u0, u1) on node 'eq'");  eq_2 = _assert_scalar_2 = None
    add_4 = torch.ops.aten.add.Tensor(getitem_5, getitem_7);  getitem_5 = getitem_7 = None
    return (getitem_6, add_4)""",
        )

    def test_compile_global(self):
        # [FUNCTION DOC] Compile a function using a global Counter. Checks values and counter
        # state across two calls, and that the counter is lifted as a graph input with the
        # mutation modeled functionally.
        counter = Counter(0, 10)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Increment the global counter twice with the op, multiply then
            # add, return x and the counter.
            z = torch.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x * z
            z = torch.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x + z
            return x, counter

        inp = (torch.tensor(1), torch.tensor(0))
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        res = opt_f(*inp)
        self.assertEqual(res[0], torch.tensor(3))
        self.assertEqual(res[1].counter, torch.tensor(2))

        res = opt_f(*inp)
        self.assertEqual(res[0], torch.tensor(7))
        self.assertEqual(res[1].counter, torch.tensor(4))

        # counter is automatically lifted as an input
        # Even though we returned counter in the eager code, it does not get
        # returned in the graph because dynamo does not detect that the object
        # is mutated.
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._TestOpaqueObject.increment_counter.default, c = arg0_1, _prev_base_index = 0, _all_bases = [arg2_1])
    getitem = auto_functionalized_v2[0]
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    mul = torch.ops.aten.mul.Tensor(arg1_1, getitem);  arg1_1 = getitem = None
    auto_functionalized_v2_1 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._TestOpaqueObject.increment_counter.default, c = arg0_1, _prev_base_index = 0, _all_bases = [getitem_1]);  arg0_1 = getitem_1 = None
    getitem_2 = auto_functionalized_v2_1[0]
    getitem_3 = auto_functionalized_v2_1[1];  auto_functionalized_v2_1 = None
    add = torch.ops.aten.add.Tensor(mul, getitem_2);  mul = getitem_2 = None
    copy_ = torch.ops.aten.copy_.default(arg2_1, getitem_3);  arg2_1 = getitem_3 = copy_ = None
    return (add,)""",
        )

    def test_compile_create_intermediate(self):
        # [FUNCTION DOC] Checks that creating an opaque object inside a compiled function causes
        # exactly one graph break whose message says the object was created mid-program.
        dynamo_counters.clear()

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Create a Counter inside the function and apply the
            # increment op.
            counter = Counter(0, 10)
            z = torch.ops._TestOpaqueObject.increment_counter(counter, y)
            x = x * z
            return x

        inp = (torch.tensor(1), torch.tensor(0))
        torch.compile(foo)(*inp)
        self.assertEqual(len(dynamo_counters["graph_break"]), 1)
        self.assertTrue(
            "An opaque object was created in the middle of the program"
            in next(iter(dynamo_counters["graph_break"].keys())),
        )

    def test_compile_attribute(self):
        # [FUNCTION DOC] Checks that accessing unregistered members (a method and a property) on a
        # reference opaque under torch.compile raises RuntimeError, and that a member registered
        # as real but returning a reference-type opaque raises the dedicated RuntimeError.
        counter = Counter(0, 10)

        def foo(counter, x):
            # [FUNCTION DOC] HELPER: Square x and call the unregistered mutating method on
            # the counter.
            x = x * x
            counter.increment_counter()
            return x

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to access unregistered member on an OpaqueObject"
        ):
            torch.compile(foo, backend="eager")(counter, torch.ones(2, 3))

        def bar(counter, x):
            # [FUNCTION DOC] HELPER: Square x and add the unregistered property of the counter.
            x = x * x
            x += counter.counter
            return x

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to access unregistered member on an OpaqueObject"
        ):
            torch.compile(bar, backend="eager")(counter, torch.ones(2, 3))

        def foo(counter, x):
            # [FUNCTION DOC] HELPER: Return the nested counter via a real-use getter (not allowed
            # to return a reference-type opaque).
            return counter.get_c()

        with self.assertRaisesRegex(
            RuntimeError,
            "Opaque object member with method-type USE_REAL returned a reference-type opaque object.",
        ):
            torch.compile(foo, backend="eager")(
                NestedCounters(Counter(1, 5)), torch.ones(2, 3)
            )

    def test_compile_fixed_stride_order(self):
        # [FUNCTION DOC] Compiles an op tagged as needing fixed stride order with a hoisted string
        # argument under inductor, and checks the result equals doubling.
        hs_name = get_opaque_type_name(HoistedString)

        torch.library.define(
            "_TestOpaqueObject::stride_op",
            f"(Tensor x, {hs_name} s) -> Tensor",
            tags=(torch.Tag.needs_fixed_stride_order,),
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::stride_op",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def stride_op_impl(x: torch.Tensor, s: HoistedString) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of stride_op: double the tensor.
            return x * 2.0

        @torch.library.register_fake("_TestOpaqueObject::stride_op", lib=self.lib)
        def stride_op_fake(x, s):
            # [FUNCTION DOC] HELPER: Fake implementation: empty tensor like x.
            return torch.empty_like(x)

        def fn(x, s):
            # [FUNCTION DOC] HELPER: Apply stride_op to x and s.
            return torch.ops._TestOpaqueObject.stride_op(x, s)

        s = HoistedString("double")
        x = torch.randn(4, 4)

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(x, s)

        expected = x * 2.0
        self.assertEqual(result, expected)

    def test_compile_exact_strides(self):
        # [FUNCTION DOC] Same as the fixed-stride test but for an op tagged as needing exact
        # strides; real op triples the tensor.
        hs_name = get_opaque_type_name(HoistedString)

        torch.library.define(
            "_TestOpaqueObject::exact_op",
            f"(Tensor x, {hs_name} s) -> Tensor",
            tags=(torch.Tag.needs_exact_strides,),
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::exact_op",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def exact_op_impl(x: torch.Tensor, s: HoistedString) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of exact_op: triple the tensor.
            return x * 3.0

        @torch.library.register_fake("_TestOpaqueObject::exact_op", lib=self.lib)
        def exact_op_fake(x, s):
            # [FUNCTION DOC] HELPER: Fake implementation: empty tensor like x.
            return torch.empty_like(x)

        def fn(x, s):
            # [FUNCTION DOC] HELPER: Apply exact_op to x and s.
            return torch.ops._TestOpaqueObject.exact_op(x, s)

        s = HoistedString("double")
        x = torch.randn(4, 4)

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(x, s)

        expected = x * 3.0
        self.assertEqual(result, expected)

    def test_compile_contiguous_strides(self):
        # [FUNCTION DOC] Same as the fixed-stride test but for an op tagged as needing contiguous
        # strides; real op multiplies the tensor by four.
        hs_name = get_opaque_type_name(HoistedString)

        torch.library.define(
            "_TestOpaqueObject::contig_op",
            f"(Tensor x, {hs_name} s) -> Tensor",
            tags=(torch.Tag.needs_contiguous_strides,),
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::contig_op",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def contig_op_impl(x: torch.Tensor, s: HoistedString) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of contig_op: multiply the tensor
            # by four.
            return x * 4.0

        @torch.library.register_fake("_TestOpaqueObject::contig_op", lib=self.lib)
        def contig_op_fake(x, s):
            # [FUNCTION DOC] HELPER: Fake implementation: empty tensor like x.
            return torch.empty_like(x)

        def fn(x, s):
            # [FUNCTION DOC] HELPER: Apply contig_op to x and s.
            return torch.ops._TestOpaqueObject.contig_op(x, s)

        s = HoistedString("double")
        x = torch.randn(4, 4)

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(x, s)

        expected = x * 4.0
        self.assertEqual(result, expected)

    def test_export_joint(self):
        # [FUNCTION DOC] Export a joint forward-backward graph for a module holding an opaque
        # AddModule and calling an op on it with an integer taken from a tensor. Defines the op,
        # its fake, and autograd formulas, checks the joint graph text, compiles it, and checks
        # the result equals eager.
        torch.library.define(
            "_TestOpaqueObject::module_mul",
            f"({get_opaque_type_name(AddModule)} a, Tensor b, SymInt c) -> Tensor",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::module_mul", "CompositeExplicitAutograd", lib=self.lib
        )
        def module_mul_impl(m: AddModule, a: torch.Tensor, b: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Real implementation of module_mul: check the type
            # (AssertionError otherwise) and call the module with the tensor and the integer.
            if not isinstance(m, AddModule):
                raise AssertionError(f"expected AddModule, got {type(m)}")
            return m(a, b)

        @torch.library.register_fake("_TestOpaqueObject::module_mul", lib=self.lib)
        def module_mul_fake(m: AddModule, a: torch.Tensor, b: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake: empty tensor like a.
            return torch.empty_like(a)

        def module_mul_setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Autograd setup: save the integer input on the context.
            m, a, b = inputs
            ctx.b = b

        def module_mul_backward(ctx, grad) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Backward: no gradient for the module, the incoming gradient
            # times the saved integer for the tensor, none for the integer.
            return None, grad * ctx.b, None

        torch.library.register_autograd(
            "_TestOpaqueObject::module_mul",
            module_mul_backward,
            setup_context=module_mul_setup_context,
            lib=self.lib,
        )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning an opaque AddModule and calling the module_mul op
            # with an integer taken from a tensor.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the parent module state and the opaque
                # AddModule attribute.
                super().__init__()
                self.moo = AddModule()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Convert y to a python number and call the module_mul op
                # on x and the number.
                b = y.item()
                return torch.ops._TestOpaqueObject.module_mul(self.moo, x, b)

        inp = (torch.randn(3, requires_grad=True), torch.tensor(4))
        with ExitStack() as stack:
            with FakeTensorMode(shape_env=ShapeEnv()):
                joint = aot_export_joint_with_descriptors(stack, M(), inp)
                self.assertExpectedInline(
                    joint.graph_module.code.strip(),
                    """\
def forward(self, primals, tangents):
    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(primals_2);  primals_2 = None
    _opaque_obj0 = self._opaque_obj0
    module_mul = torch.ops._TestOpaqueObject.module_mul.default(_opaque_obj0, primals_1, _local_scalar_dense);  _opaque_obj0 = primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, _local_scalar_dense);  tangents_1 = _local_scalar_dense = None
    return pytree.tree_unflatten([module_mul, mul_1, None], self._out_spec)""",
                )
                compiled_fn = aot_compile_joint_with_descriptors(joint)

        self.assertEqual(compiled_fn(*inp), M()(*inp))

    def test_invalid_reference_type(self):
        # [FUNCTION DOC] Checks registering a class that does not derive from the opaque base
        # raises TypeError; and that a class registered with a member name that does not
        # exist, when used under torch.compile, raises an internal dynamo error naming the
        # type and the member.
        # Test that classes without subclassing CustomClassBase are rejected
        class NoOpaqueBase:
            # [CLASS DOC] HELPER: Plain class without the opaque base, used to check registration
            # is rejected.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store x.
                self.x = x

        with self.assertRaisesRegex(
            TypeError,
            "must subclass torch._custom_class_base.CustomClassBase",
        ):
            register_custom_class(NoOpaqueBase, typ="symbolic")

        class BadMember(CustomClassBase):
            # [CLASS DOC] HELPER: Opaque class with an attribute x that is registered with a
            # different, missing member name to trigger the missing-member error.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store x.
                self.x = x

        def foo(bad, y):
            # [FUNCTION DOC] HELPER: Compiled body adding the attribute x of the object to y.
            return y + bad.x

        register_custom_class(
            BadMember, typ="symbolic", members={"y": MemberType.USE_REAL}
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            f"Opaque object of type '{get_opaque_type_name(BadMember)}' was specified to have member 'y'",
        ):
            torch.compile(foo)(BadMember(1), torch.ones(1))

    def test_invalid_value_type(self):
        # [FUNCTION DOC] Checks registration validation of constant opaque types: each of the
        # missing equality, missing hash, missing FX representation, and the (unneeded) guard
        # function being given raises TypeError with a specific message.
        class NoEq(CustomClassBase):
            # [CLASS DOC] HELPER: Opaque class with neither custom equality nor hash.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store x.
                self.x = x

        with self.assertRaisesRegex(
            TypeError, "expected to have a non-default `__eq__`"
        ):
            register_custom_class(NoEq, typ="constant")

        class NoHash(CustomClassBase):
            # [CLASS DOC] HELPER: Opaque class with equality but no hash.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store x.
                self.x = x

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compare x of both objects.
                return self.x == other.x

        with self.assertRaisesRegex(
            TypeError, "expected to have a non-default `__hash__`"
        ):
            register_custom_class(NoHash, typ="constant")

        class NoRepr(CustomClassBase):
            # [CLASS DOC] HELPER: Opaque class with equality and hash but no FX representation.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store x.
                self.x = x

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compare x of both objects.
                return self.x == other.x

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash of x.
                return hash(self.x)

        with self.assertRaisesRegex(TypeError, "expected to have a `__fx_repr__`"):
            register_custom_class(NoRepr, typ="constant")

        class SpecifyMember(CustomClassBase):
            # [CLASS DOC] HELPER: Valid constant opaque class (equality, hash, FX repr),
            # registered with a guard function to trigger the "no need to specify" error.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store x.
                self.x = x

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compare x of both objects.
                return self.x == other.x

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash of x.
                return hash(self.x)

            def __fx_repr__(self):
                # [FUNCTION DOC] HELPER: Return only the text form; a plain string rather than the
                # expected pair.
                # KNOWN ISSUE: returns a bare string instead of the (text, globals mapping)
                #     pair used elsewhere; safe only because registration fails before it is
                #     ever called.
                return f"SpecifyMember({self.x})"

        with self.assertRaisesRegex(TypeError, "No need to specify `guard_fn`"):
            register_custom_class(
                SpecifyMember, typ="constant", guard_fn=lambda obj: []
            )

    def test_invalid_schema(self):
        # [FUNCTION DOC] Checks that defining ops with malformed schema strings (unknown dotted
        # type, stray dots, doubled separators) raises RuntimeError with parser messages.
        with self.assertRaisesRegex(
            RuntimeError,
            "unknown type specifier",
        ):
            torch.library.define(
                "_TestOpaqueObject::invalid_op1",
                "(foo.bar.baz a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"expected \) but found 'dots' here",
        ):
            torch.library.define(
                "_TestOpaqueObject::invalid_op2",
                "(......... a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "unknown type specifier",
        ):
            torch.library.define(
                "_TestOpaqueObject::invalid_op5",
                "(MyNamespace..MyClass a) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=self.lib,
            )

    def test_invalid_opaque_obj_types(self):
        # [FUNCTION DOC] Checks that registering built-in types as opaque raises ValueError, that
        # a class already registered as a pytree node cannot become opaque, and that a class
        # registered opaque cannot be registered as a pytree. Restores the pytree registries in a
        # finally block.
        for t in [str, bool, int, float, torch.Tensor]:
            with self.assertRaisesRegex(ValueError, "Unable to register built-in type"):
                register_custom_class(t, typ="symbolic")

        @dataclass
        class Bad1(CustomClassBase):
            # [CLASS DOC] HELPER: Dataclass with an opaque base that is registered as a pytree
            # dataclass first.
            x: int

        pytree.register_dataclass(Bad1)
        try:
            with self.assertRaisesRegex(
                ValueError,
                "cannot be registered as an opaque object as it has been registered as a pytree.",
            ):
                register_custom_class(Bad1, typ="symbolic")
        finally:
            # Clean up pytree registration to avoid leaking state to other tests
            pytree.SUPPORTED_NODES.pop(Bad1, None)
            pytree.SUPPORTED_SERIALIZED_TYPES.pop(Bad1, None)
            pytree.CONSTANT_NODES.discard(Bad1)

        @dataclass
        class Bad2(CustomClassBase):
            # [CLASS DOC] HELPER: Dataclass with an opaque base registered opaque first, then
            # as a pytree.
            x: int

        register_custom_class(Bad2, typ="symbolic")
        with self.assertRaisesRegex(
            ValueError,
            "cannot be registered as a pytree as it has been registered as an opaque object.",
        ):
            pytree.register_dataclass(Bad2)

    def test_value_type_recompile(self):
        # [FUNCTION DOC] Checks value-type guards: compile counts show same-valued configs reuse
        # the graph, while a different mode causes one recompile, and results match the ops.
        cnt = CompileCounter()

        def foo(x, cfg):
            # [FUNCTION DOC] HELPER: Apply process_with_config to x and cfg.
            return torch.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = torch.randn(3, 3)

        opt_f = torch.compile(foo, backend=cnt, fullgraph=True)
        res = opt_f(x, ValueConfig("square"))
        self.assertEqual(res, x * x)
        self.assertEqual(cnt.frame_count, 1)

        res = opt_f(x, ValueConfig("square"))
        self.assertEqual(res, x * x)
        self.assertEqual(cnt.frame_count, 1)

        # Recompilation!
        res = opt_f(x, ValueConfig("double"))
        self.assertEqual(res, x + x)
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_value_type_graph_output(self, backend):
        # [FUNCTION DOC] Parametrized over eager, aot_eager and inductor backends. Compiles a
        # function returning a tensor and a constructed value-type config, checks the returned
        # config is equal, and that dynamo export capture also returns an equal config.
        def foo(x):
            # [FUNCTION DOC] HELPER: Return x squared and a new square-mode config.
            return x * x, ValueConfig("square")

        x = torch.randn(3, 3)
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        res = opt_f(x)
        self.assertEqual(res[1], ValueConfig("square"))

        gm = _dynamo_graph_capture_for_export(foo)(x)
        res = gm(x)
        self.assertEqual(res[1], ValueConfig("square"))

    def test_value_type_graph_output_subclass_metadata_side_effect(self):
        # [FUNCTION DOC] Checks a value-type opaque created inside compiled code, stored in an
        # outside dict and put as metadata of a returned wrapper subclass, keeps its type and mode
        # both in the output and in the dict.
        class TensorWithValueMetadata(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass with one inner data tensor and a
            # value-type config carried as flatten metadata; dispatch unwraps and runs the op
            # without re-wrapping.
            @staticmethod
            def __new__(cls, data, config):
                # [FUNCTION DOC] HELPER: Create the wrapper tensor from the data metadata and
                # attach data and config.
                t = torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.shape,
                    strides=data.stride(),
                    dtype=data.dtype,
                    device=data.device,
                    layout=data.layout,
                    requires_grad=data.requires_grad,
                )
                t._data = data
                t._config = config
                return t

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook: the inner data name and a metadata mapping
                # holding the config.
                return ["_data"], {"config": self._config}

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return a fixed placeholder string.
                return "TensorWithValueMetadata(...)"

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Unflatten hook: rebuild from the inner data and the
                # config in the metadata.
                return TensorWithValueMetadata(inner_tensors["_data"], meta["config"])

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook: unwrap subclass args and run the op,
                # returning plain results.
                def unwrap(x):
                    # [FUNCTION DOC] HELPER: Return the inner data for a subclass instance,
                    # otherwise the value.
                    if isinstance(x, TensorWithValueMetadata):
                        return x._data
                    return x

                return func(
                    *pytree.tree_map(unwrap, args),
                    **pytree.tree_map(unwrap, kwargs or {}),
                )

        store = {}

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: create a config, record it in the outer dict,
            # and return a wrapper subclass of x plus one with it.
            config = ValueConfig("square")
            store["config"] = config
            return TensorWithValueMetadata(x + 1, config)

        out = fn(torch.zeros(2))

        self.assertIs(type(out._config), ValueConfig)
        self.assertEqual(out._config.mode, "square")
        self.assertIs(type(store["config"]), ValueConfig)
        self.assertEqual(store["config"].mode, "square")

    def test_value_type_graph_input(self):
        # [FUNCTION DOC] Checks that a value-type config passed as input is embedded as a literal
        # in the graph (not a graph input), with a separate graph for each different mode; checks
        # graph texts.
        # Even though cfg is an input, it should not be an input to the dynamo
        # graph. Instead it should directly put in the graph argument as a
        # constant.

        def foo(x, cfg):
            # [FUNCTION DOC] HELPER: Apply process_with_config to x and cfg.
            return torch.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = torch.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, ValueConfig("square"))

        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    process_with_config = torch.ops._TestOpaqueObject.process_with_config(l_x_, ValueConfig(mode='square'));  l_x_ = None
    return (process_with_config,)""",
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='square'));  arg0_1 = None
    return (process_with_config,)""",
        )

        opt_f(x, ValueConfig("double"))

        self.assertExpectedInline(
            backend.fw_graphs[1].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='double'));  arg0_1 = None
    return (process_with_config,)""",
        )

    def test_value_type_graph_intermediate(self):
        # [FUNCTION DOC] Checks a config constructed inside compiled code from a string argument
        # is embedded as a literal in the dynamo and AOT graphs, one graph per string, and that
        # inductor matches eager for both modes.
        def foo(x, config):
            # [FUNCTION DOC] HELPER: Build a config from the string and apply process_with_config.
            cfg = ValueConfig(config)
            return torch.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = torch.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, "square")

        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    process_with_config = torch.ops._TestOpaqueObject.process_with_config(l_x_, ValueConfig(mode='square'));  l_x_ = None
    return (process_with_config,)""",
        )
        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='square'));  arg0_1 = None
    return (process_with_config,)""",
        )

        opt_f(x, "double")
        self.assertExpectedInline(
            backend.fw_graphs[1].code.strip(),
            """\
def forward(self, arg0_1):
    process_with_config = torch.ops._TestOpaqueObject.process_with_config.default(arg0_1, ValueConfig(mode='double'));  arg0_1 = None
    return (process_with_config,)""",
        )

        opt_f = torch.compile(foo, fullgraph=True, backend="inductor")
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, "square"), foo(x, "square"))
        self.assertEqual(opt_f(x, "double"), foo(x, "double"))

    def test_value_type_attr_access(self):
        # [FUNCTION DOC] Checks attribute access and method calls on a value type created in
        # compiled code are folded into constants: the AOT graph contains no opaque-related nodes,
        # only constant values.
        def foo(x):
            # [FUNCTION DOC] HELPER: Create a size store from the shape, build ones and
            # concatenate, call the increment method (result discarded) and add the size.
            size = SizeStore(x.shape[0])
            t1 = torch.ones((size.size,))
            t2 = torch.cat([x, t1])
            size.increment_size()
            return t2 + size.size

        x = torch.randn(3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        res = opt_f(x)
        self.assertEqual(res, foo(x))

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([3], device = device(type='cpu'), pin_memory = False)
    cat = torch.ops.aten.cat.default([arg0_1, ones]);  arg0_1 = ones = None
    add = torch.ops.aten.add.Tensor(cat, 3);  cat = None
    return (add,)""",
        )

    def test_value_type_unregistered_method(self):
        # [FUNCTION DOC] Checks an unregistered method-like use on a value type (string length of
        # the mode) is inlined as a constant without error.
        # Unregistered methods on value types should inline (no error)
        def foo(x):
            # [FUNCTION DOC] HELPER: Add the length of the mode string of a new config to x.
            cfg = ValueConfig("square")
            return x + len(cfg.mode)

        x = torch.randn(3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        res = opt_f(x)
        self.assertEqual(res, foo(x))

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 6);  arg0_1 = None
    return (add,)""",
        )

    def test_weakref_cleanup(self):
        # [FUNCTION DOC] Checks that after the only reference to a locally defined opaque class is
        # dropped and garbage collected, no registry entry mentions it any more.
        def register_tmp_class():
            # [FUNCTION DOC] HELPER: Define a temporary constant opaque class, register it, check
            # it is recognized as custom, constant and in the registry, and return its type name.
            class TmpClass(CustomClassBase):
                # [CLASS DOC] HELPER: Temporary constant opaque class with a value, equality, hash
                # and FX repr.
                def __init__(self, value):
                    # [FUNCTION DOC] HELPER: Store the value.
                    self.value = value

                def __eq__(self, other):
                    # [FUNCTION DOC] HELPER: Compare values.
                    return self.value == other.value

                def __hash__(self):
                    # [FUNCTION DOC] HELPER: Hash of the value.
                    return hash(self.value)

                def __fx_repr__(self):
                    # [FUNCTION DOC] HELPER: Return the pair of constructor text with the value
                    # and the class mapping.
                    return f"TmpClass(value={self.value!r})", {"TmpClass": TmpClass}

            register_custom_class(TmpClass, typ="constant")

            self.assertTrue(is_custom_class(TmpClass))
            self.assertTrue(is_opaque_constant_type(TmpClass))
            self.assertIn(TmpClass, _OPAQUE_TYPES)

            return get_opaque_type_name(TmpClass)

        # registers TmpClass as opaque
        tmp_class_name = register_tmp_class()

        # garbage collect TmpClass
        gc.collect()

        # Verify that the class is no longer registered
        for opaque_info in _OPAQUE_TYPES.values():
            self.assertFalse(tmp_class_name in opaque_info.class_name)

    def test_value_type_nested(self):
        # [FUNCTION DOC] Checks nested value types built in compiled code appear as one nested
        # literal in the AOT graph and that inductor matches eager for two modes.
        def foo(x, config):
            # [FUNCTION DOC] HELPER: Build a nested config from the shape size and a mode, apply
            # the nested op.
            size = SizeStore(x.shape[0])
            cfg = NestedValueSize(size, ValueConfig(config))
            return torch.ops._TestOpaqueObject.process_nested_config(x, cfg)

        x = torch.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        opt_f(x, "square")

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_nested_config = torch.ops._TestOpaqueObject.process_nested_config.default(arg0_1, NestedValueSize(size=SizeStore(size=3), config=ValueConfig(mode='square')));  arg0_1 = None
    return (process_nested_config,)""",
        )

        opt_f = torch.compile(foo, fullgraph=True, backend="inductor")
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, "square"), foo(x, "square"))
        self.assertEqual(opt_f(x, "double"), foo(x, "double"))

    def test_value_type_list(self):
        # [FUNCTION DOC] Checks a list of value-type objects built in compiled code is passed as a
        # list literal in the AOT graph, and that inductor matches eager.
        def foo(x):
            # [FUNCTION DOC] HELPER: Make a size store per dimension of x and apply the multi-
            # size op.
            sizes = []
            for size in x.size():
                sizes.append(SizeStore(size))
            return torch.ops._TestOpaqueObject.process_multiple_sizes(x, sizes)

        x = torch.randn(3, 3)
        backend = AotEagerAndRecordGraphs()
        opt_f = torch.compile(foo, fullgraph=True, backend=backend)
        opt_f(x)

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1):
    process_multiple_sizes = torch.ops._TestOpaqueObject.process_multiple_sizes.default(arg0_1, [SizeStore(size=3), SizeStore(size=3)]);  arg0_1 = None
    return (process_multiple_sizes,)""",
        )

        opt_f = torch.compile(foo, fullgraph=True, backend="inductor")
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x), foo(x))

    def test_getitem(self):
        # [FUNCTION DOC] Checks that an indexing member on an opaque counters object stored in
        # a tensor subclass is traced into the dynamo graph as a call, and the value at index
        # one is used.
        def foo(x):
            # [FUNCTION DOC] HELPER: Double and increment x, fetch its counters, index them using
            # the size store's size, and multiply by the selected counter's start tensor.
            y = x * 2 + 1
            nested_counter = y.get_counter()
            s = y.size_store
            counter = nested_counter[s.size]
            return y * counter.get_start_tensor()

        x = TensorWithCounter(
            torch.tensor(1),
            torch.tensor(1),
            NestedCounters([Counter(1, 5), Counter(2, 5)]),
            SizeStore(1),
        )

        inp = (x,)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(foo, fullgraph=True, backend=backend)(*inp)
        self.assertEqual(res, torch.tensor(6))

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "TensorWithCounter(i64[])"):
        l_x_ = L_x_

        mul: "TensorWithCounter(i64[])" = l_x_ * 2;  l_x_ = None
        add: "TensorWithCounter(i64[])" = mul + 1;  mul = None

        get_counter = add.get_counter()

        getitem = get_counter.__getitem__(1);  get_counter = None

        get_start_tensor: "i64[]" = getitem.get_start_tensor();  getitem = None
        mul_1: "TensorWithCounter(i64[])" = add * get_start_tensor;  add = get_start_tensor = None
        return (mul_1,)
""",
        )

    def test_tensor_subclass_with_opaque_attr(self):
        # [FUNCTION DOC] Checks that opaque attributes of a tensor subclass are accessed in the
        # traced graph (getters kept, literals folded) and that a changed counter or size store
        # each cause a recompile.
        def fn(x):
            # [FUNCTION DOC] HELPER: Read the counter and size store through attributes and
            # getters and fold their numbers into arithmetic with the tensor.
            y = x * 2 + 1
            c1 = y._counter
            c2 = y.get_counter()
            s1 = y.size_store
            s2 = y.get_size_store()
            return y * c1.start + c2.start + s1.size + s2.size

        a = torch.rand(4, 4)
        b = torch.rand(4, 4)
        counter = Counter(start=3, end=10)
        size = SizeStore(4)
        x = TensorWithCounter(a, b, counter, size)

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        opt_fn(x)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "TensorWithCounter(f32[4, 4])"):
        l_x_ = L_x_

        mul: "TensorWithCounter(f32[4, 4])" = l_x_ * 2;  l_x_ = None
        add: "TensorWithCounter(f32[4, 4])" = mul + 1;  mul = None

        get_counter = add.get_counter();  get_counter = None

        get_size_store = add.get_size_store();  get_size_store = None

        getattr_1 = add._counter;  getattr_1 = None

        mul_1: "TensorWithCounter(f32[4, 4])" = add * 3;  add = None
        add_1: "TensorWithCounter(f32[4, 4])" = mul_1 + 3;  mul_1 = None
        add_2: "TensorWithCounter(f32[4, 4])" = add_1 + 4;  add_1 = None
        add_3: "TensorWithCounter(f32[4, 4])" = add_2 + 4;  add_2 = None
        return (add_3,)
""",
        )
        self.assertEqual(cnt.frame_count, 1)

        a = torch.rand(4, 4)
        b = torch.rand(4, 4)
        counter = Counter(start=1, end=10)
        x = TensorWithCounter(a, b, counter, SizeStore(4))
        opt_fn(x)

        # Recompile since Counter has changed
        self.assertEqual(cnt.frame_count, 2)

        a = torch.rand(4, 4)
        b = torch.rand(4, 4)
        counter = Counter(start=1, end=10)
        x = TensorWithCounter(a, b, counter, SizeStore(5))
        opt_fn(x)

        # Recompile since SizeStore has changed
        self.assertEqual(cnt.frame_count, 3)

    def test_tensor_subclass_with_opaque_attr_backward(self):
        # [FUNCTION DOC] Compile a simple function on a TensorWithCounter with AOT eager, check
        # the output keeps the very same counter and size store objects as the input, then run
        # backward and check the gradient is also a subclass. A second call with equal-but-
        # distinct opaque instances must not recompile, and outputs and gradients must carry the
        # new runtime counter instance (identity), while the value-type size store is only
        # compared by equality.
        """Test opaque objects in tensor subclass are correctly remapped through backward."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x doubled plus one.
            return x * 2 + 1

        a = torch.rand(4, 4, requires_grad=True)
        b = torch.rand(4, 4, requires_grad=True)
        counter = Counter(start=3, end=10)
        size = SizeStore(4)
        x = TensorWithCounter(a, b, counter, size)

        cnt = CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        out = opt_fn(x)

        # Output should be TensorWithCounter with same opaque objects
        self.assertTrue(isinstance(out, TensorWithCounter))
        self.assertIs(out._counter, counter)
        self.assertIs(out._size_store, size)

        # Run backward
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(isinstance(x.grad, TensorWithCounter))

        self.assertEqual(cnt.frame_count, 1)

        # Call again with different instances that are __eq__ equal.
        # Metadata guard uses ==, so equal values pass the guard.
        # But they're different objects, so assertIs tests the fix.
        a2 = torch.rand(4, 4, requires_grad=True)
        b2 = torch.rand(4, 4, requires_grad=True)
        counter2 = Counter(start=3, end=10)  # Same values, different instance
        size2 = SizeStore(4)  # Same value, different instance
        self.assertEqual(counter, counter2)  # Verify they're equal
        self.assertIsNot(counter, counter2)  # But not the same object
        x2 = TensorWithCounter(a2, b2, counter2, size2)

        out2 = opt_fn(x2)
        self.assertEqual(cnt.frame_count, 1)  # No recompilation

        self.assertTrue(isinstance(out2, TensorWithCounter))
        # Key checks: should use runtime instances, not traced instances
        self.assertIs(out2._counter, counter2)
        # Value-type opaques in ctx are baked at compile time
        self.assertEqual(out2._size_store, size2)

        out2.sum().backward()
        self.assertIsNotNone(x2.grad)
        self.assertTrue(isinstance(x2.grad, TensorWithCounter))
        self.assertIs(x2.grad._counter, counter2)
        self.assertEqual(x2.grad._size_store, size2)

    def test_tangent_primal_proxy_collision_for_opaque_inner_attr(self):
        # [FUNCTION DOC] Regression test, no compilation: two fake wrappers around the same real
        # counter (as primal and tangent in joint tracing) get proxies registered one after the
        # other on a graph tracer; asserts that both wrappers resolve to the primal proxy node.
        """Regression test for tangent/primal proxy collision.

        When a tensor subclass has an opaque inner attr, joint graph tracing
        creates separate FakeScriptObject wrappers for the primal and tangent
        that share the same underlying real object.  set_proxy_slot must map
        the tangent wrapper to the *primal* proxy so that forward outputs
        don't spuriously depend on tangent placeholders (which would crash the
        partitioner with 'Node tangents_N was invalid, but is output').
        """
        from torch._library.fake_class_registry import FakeScriptObject
        from torch.fx.experimental.proxy_tensor import (
            _GraphAppendingTracerEx,
            set_proxy_slot,
        )

        counter = Counter(start=3, end=10)
        fso_primal = FakeScriptObject(counter, "Counter", counter)
        fso_tangent = FakeScriptObject(counter, "Counter", counter)
        # Sanity: different wrappers, same real_obj
        self.assertIsNot(fso_primal, fso_tangent)
        self.assertIs(
            object.__getattribute__(fso_primal, "real_obj"),
            object.__getattribute__(fso_tangent, "real_obj"),
        )

        graph = torch.fx.Graph()
        tracer = _GraphAppendingTracerEx(graph)

        primal_node = graph.placeholder("primals_1")
        tangent_node = graph.placeholder("tangents_1")
        primal_proxy = torch.fx.Proxy(primal_node, tracer)
        tangent_proxy = torch.fx.Proxy(tangent_node, tracer)

        # Register primal first, then tangent (mirrors joint graph tracing)
        set_proxy_slot(fso_primal, tracer, primal_proxy)
        set_proxy_slot(fso_tangent, tracer, tangent_proxy)

        # Both wrappers should resolve to the primal proxy
        self.assertIs(tracer.opaque_tracker[fso_primal].node, primal_node)
        self.assertIs(tracer.opaque_tracker[fso_tangent].node, primal_node)

    def test_opaque_produced_by_call_function_saved_for_backward(self):
        # [FUNCTION DOC] Regression test: an opaque produced by an op inside the graph (not an
        # input) must be savable for backward by the partitioner. Compiles a function creating a
        # multiplier from a tensor and applying the scaling op, then checks the output and the
        # gradient equal 2.5 scaling.
        """Test that an opaque object produced by a call_function node
        (not a placeholder) is correctly saved for backward.

        Without is_opaque_node() in the min_cut partitioner, this crashes
        with 'AOT Autograd failed to partition' because the opaque node
        is a non-tensor call_function that can't be saved or recomputed."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Create a multiplier opaque from a constant scale tensor and
            # apply the mul_with_scale op to x.
            scale = torch.tensor(2.5)
            multiplier = torch.ops._TestOpaqueObject.create_multiplier(scale)
            return torch.ops._TestOpaqueObject.mul_with_scale(multiplier, x)

        x = torch.randn(3, 3, requires_grad=True)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        out = opt_fn(x)
        self.assertTrue(torch.allclose(out, x * 2.5))

        out.backward(torch.ones_like(out))
        self.assertIsNotNone(x.grad)
        expected_grad = torch.ones_like(x) * 2.5
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_tensor_subclass_opaque_backward_compiled_autograd(self):
        # [FUNCTION DOC] Compile a multiply on a TensorWithCounter with AOT eager and run backward
        # under compiled autograd, twice (second time with equal but distinct opaque instances).
        # Checks outputs and gradients are subclass instances and that the counter objects are
        # preserved by identity.
        """Test opaque objects work with compiled autograd backward."""
        import torch._dynamo.compiled_autograd

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x doubled.
            return x * 2

        a = torch.rand(4, 4, requires_grad=True)
        b = torch.rand(4, 4, requires_grad=True)
        counter = Counter(start=5, end=10)
        size = SizeStore(4)
        x = TensorWithCounter(a, b, counter, size)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        out = opt_fn(x)

        self.assertTrue(isinstance(out, TensorWithCounter))
        self.assertIs(out._counter, counter)
        self.assertIs(out._size_store, size)

        with torch._dynamo.compiled_autograd._enable(torch.compile(backend="eager")):
            out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(isinstance(x.grad, TensorWithCounter))

        # Second invocation with different-but-equal opaques
        a2 = torch.rand(4, 4, requires_grad=True)
        b2 = torch.rand(4, 4, requires_grad=True)
        counter2 = Counter(start=5, end=10)
        size2 = SizeStore(4)
        x2 = TensorWithCounter(a2, b2, counter2, size2)

        out2 = opt_fn(x2)
        self.assertTrue(isinstance(out2, TensorWithCounter))
        self.assertIs(out2._counter, counter2)
        self.assertEqual(out2._size_store, size2)

        with torch._dynamo.compiled_autograd._enable(torch.compile(backend="eager")):
            out2.sum().backward()

        self.assertIsNotNone(x2.grad)
        self.assertTrue(isinstance(x2.grad, TensorWithCounter))
        self.assertIs(x2.grad._counter, counter2)
        self.assertEqual(x2.grad._size_store, size2)

    def test_tensor_subclass_shared_opaque_remapping(self):
        # [FUNCTION DOC] Two subclass inputs share the same counter and size store; compile an
        # addition, check output holds the shared counter; repeat with new equal shared instances
        # and check the output holds those.
        """Test that shared opaques across inputs are correctly handled."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the sum of both inputs.
            return x + y

        counter = Counter(start=3, end=10)  # Shared opaque
        size = SizeStore(4)

        a1 = torch.rand(4, 4, requires_grad=True)
        b1 = torch.rand(4, 4, requires_grad=True)
        x = TensorWithCounter(a1, b1, counter, size)

        a2 = torch.rand(4, 4, requires_grad=True)
        b2 = torch.rand(4, 4, requires_grad=True)
        y = TensorWithCounter(a2, b2, counter, size)  # Same counter

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        out = opt_fn(x, y)

        self.assertTrue(isinstance(out, TensorWithCounter))
        self.assertIs(out._counter, counter)

        # Second invocation with different-but-equal shared opaques
        counter2 = Counter(start=3, end=10)
        size2 = SizeStore(4)

        a3 = torch.rand(4, 4, requires_grad=True)
        b3 = torch.rand(4, 4, requires_grad=True)
        x2 = TensorWithCounter(a3, b3, counter2, size2)

        a4 = torch.rand(4, 4, requires_grad=True)
        b4 = torch.rand(4, 4, requires_grad=True)
        y2 = TensorWithCounter(a4, b4, counter2, size2)

        out2 = opt_fn(x2, y2)
        self.assertTrue(isinstance(out2, TensorWithCounter))
        self.assertIs(out2._counter, counter2)
        self.assertEqual(out2._size_store, size2)

    def test_shared_opaque_identity_guard(self):
        # [FUNCTION DOC] Compile addition of two subclass inputs sharing an opaque, in the order y
        # plus x. Same sharing pattern must not recompile; breaking the sharing must recompile
        # (frame count two) and the output must carry y's counter (checked by identity).
        """When inputs share an opaque at trace time, breaking that sharing
        should trigger recompilation.

        Uses y + x so __torch_dispatch__ picks y's opaque (input 1), but
        first-occurrence-wins remapping points to x's position (input 0).
        With broken sharing, the output silently gets the wrong opaque.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return y plus x.
            return y + x

        counter = Counter(start=3, end=10)
        size = SizeStore(4)

        x = TensorWithCounter(torch.rand(4, 4), torch.rand(4, 4), counter, size)
        y = TensorWithCounter(torch.rand(4, 4), torch.rand(4, 4), counter, size)

        cnt = CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        out = opt_fn(x, y)
        self.assertEqual(cnt.frame_count, 1)
        # Shared opaques: output should have the shared counter
        self.assertIs(out._counter, counter)

        # Same sharing pattern → no recompile
        counter2 = Counter(start=3, end=10)
        size2 = SizeStore(4)
        x2 = TensorWithCounter(torch.rand(4, 4), torch.rand(4, 4), counter2, size2)
        y2 = TensorWithCounter(torch.rand(4, 4), torch.rand(4, 4), counter2, size2)
        out2 = opt_fn(x2, y2)
        self.assertEqual(cnt.frame_count, 1)
        self.assertIs(out2._counter, counter2)

        # Sharing broken → should recompile.
        # __torch_dispatch__ picks y's opaque, so the output should have
        # counter3b. Without the guard, the remapping uses x's position
        # and the output silently gets counter3a.
        counter3a = Counter(start=3, end=10)
        counter3b = Counter(start=3, end=10)
        size3 = SizeStore(4)
        x3 = TensorWithCounter(torch.rand(4, 4), torch.rand(4, 4), counter3a, size3)
        y3 = TensorWithCounter(torch.rand(4, 4), torch.rand(4, 4), counter3b, size3)
        out3 = opt_fn(x3, y3)
        self.assertIs(out3._counter, counter3b)
        self.assertEqual(cnt.frame_count, 2)

    def test_shared_direct_opaque_identity_guard(self):
        # [FUNCTION DOC] Pass the same counter object as two direct arguments to an op-calling
        # function. Same sharing again does not recompile; distinct but equal counters recompile
        # because an identity guard is needed (frame count two). Results are checked numerically
        # each time.
        """When the same opaque is passed as multiple direct function
        arguments, breaking the sharing should trigger recompilation.

        Unlike test_shared_opaque_identity_guard where opaques are inside
        tensor subclass metadata, here opaques are direct function arguments.
        The tracer may conflate them when they're the same object, causing
        counter_start(counter_b) to return counter_a's start instead.
        """
        get_start = torch.ops._TestOpaqueObject.counter_start

        def fn(counter_a, counter_b, x):
            # [FUNCTION DOC] HELPER: Return the start tensor of each counter plus x, as a pair.
            return get_start(counter_a) + x, get_start(counter_b) + x

        counter = Counter(start=3, end=10)
        x = torch.rand(4, 4)

        cnt = CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        out_a, out_b = opt_fn(counter, counter, x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(out_a, 3 + x)
        self.assertEqual(out_b, 3 + x)

        # Same sharing → no recompile
        counter2 = Counter(start=3, end=10)
        out_a2, out_b2 = opt_fn(counter2, counter2, x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(out_a2, 3 + x)
        self.assertEqual(out_b2, 3 + x)

        # Broken sharing → should recompile.
        # Both counters have start=3 so the existing guard_fn doesn't catch
        # the difference — only an identity guard would.
        counter3a = Counter(start=3, end=10)
        counter3b = Counter(start=3, end=10)
        out_a3, out_b3 = opt_fn(counter3a, counter3b, x)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(out_a3, 3 + x)
        self.assertEqual(out_b3, 3 + x)

    def test_tensor_subclass_shared_opaque_backward(self):
        # [FUNCTION DOC] Backward with two subclass inputs sharing the same opaques: compile
        # addition, run backward, check both inputs get gradients; repeat with new shared
        # instances and check the output and gradient carry the new counter by identity and the
        # equal size store.
        """Test backward with multiple subclass inputs sharing the same opaque.

        This tests the case where grad_inputs share an opaque object.
        The bug was that create_subclass_meta(grad_inputs) would incorrectly
        build opaque_remappings between grad_inputs, causing an assertion
        failure in wrap_tensor_subclasses.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the sum of both inputs.
            return x + y

        # Create two TensorWithCounter inputs sharing the SAME opaque objects
        counter = Counter(start=3, end=10)
        size = SizeStore(4)

        a1 = torch.rand(4, 4, requires_grad=True)
        b1 = torch.rand(4, 4, requires_grad=True)
        x = TensorWithCounter(a1, b1, counter, size)

        a2 = torch.rand(4, 4, requires_grad=True)
        b2 = torch.rand(4, 4, requires_grad=True)
        y = TensorWithCounter(a2, b2, counter, size)  # Same counter and size

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        out = opt_fn(x, y)

        self.assertTrue(isinstance(out, TensorWithCounter))
        self.assertIs(out._counter, counter)

        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

        # Second invocation with different-but-equal shared opaques
        counter2 = Counter(start=3, end=10)
        size2 = SizeStore(4)

        a3 = torch.rand(4, 4, requires_grad=True)
        b3 = torch.rand(4, 4, requires_grad=True)
        x2 = TensorWithCounter(a3, b3, counter2, size2)

        a4 = torch.rand(4, 4, requires_grad=True)
        b4 = torch.rand(4, 4, requires_grad=True)
        y2 = TensorWithCounter(a4, b4, counter2, size2)

        out2 = opt_fn(x2, y2)
        self.assertTrue(isinstance(out2, TensorWithCounter))
        self.assertIs(out2._counter, counter2)
        self.assertEqual(out2._size_store, size2)

        out2.sum().backward()
        self.assertIsNotNone(x2.grad)
        self.assertIsNotNone(y2.grad)
        self.assertTrue(isinstance(x2.grad, TensorWithCounter))
        self.assertIs(x2.grad._counter, counter2)
        self.assertEqual(x2.grad._size_store, size2)

    def test_deeply_nested_tensor_subclass_with_opaque(self):
        # [FUNCTION DOC] Compile an arithmetic function on a nested wrapper (outer subclass
        # holding a TensorWithCounter) with AOT eager. Checks the output keeps the nested
        # structure, scale and the same counter and size store objects; backward works and gives
        # nested gradients; a second call with equal-but-distinct instances does not recompile and
        # carries the new counter instance by identity.
        """Test opaque objects in nested tensor subclasses are correctly remapped."""

        # Outer subclass that wraps TensorWithCounter
        class NestedSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Outer wrapper tensor subclass holding an inner TensorWithCounter
            # and a scale value, with metadata taken from the inner tensor.
            @staticmethod
            def __new__(cls, inner, scale):
                # [FUNCTION DOC] HELPER: Create the wrapper tensor with the inner tensor's size,
                # strides, offset, device, layout, dtype and requires_grad.
                # inner is a TensorWithCounter
                kwargs = {
                    "strides": inner.stride(),
                    "storage_offset": inner.storage_offset(),
                    "device": inner.device,
                    "layout": inner.layout,
                    "requires_grad": inner.requires_grad,
                    "dtype": inner.dtype,
                }
                return torch.Tensor._make_wrapper_subclass(cls, inner.size(), **kwargs)

            def __init__(self, inner, scale):
                # [FUNCTION DOC] HELPER: Store the inner tensor and the scale.
                self._inner = inner
                self._scale = scale

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return a string showing the inner tensor and the scale.
                return f"NestedSubclass({self._inner}, scale={self._scale})"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook: the inner attribute name and a one-element
                # context tuple with the scale.
                return ["_inner"], (self._scale,)

            @staticmethod
            def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Unflatten hook: rebuild the outer wrapper from the inner
                # mapping and the scale in the context tuple.
                (scale,) = ctx
                return NestedSubclass(inner_tensors["_inner"], scale)

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch hook: finds the scale from the first outer-
                # wrapper arg, unwraps to the inner tensors, runs the op, and re-wraps any inner-
                # subclass results into the outer wrapper with the same scale.
                if kwargs is None:
                    kwargs = {}

                def unwrap(x):
                    # [FUNCTION DOC] HELPER: Return the inner tensor for an outer wrapper, else
                    # the value unchanged.
                    return x._inner if isinstance(x, NestedSubclass) else x

                def get_scale(x):
                    # [FUNCTION DOC] HELPER: Return the scale of an outer wrapper, or None for
                    # anything else.
                    if isinstance(x, NestedSubclass):
                        return x._scale
                    return None

                scale = None
                for arg in torch.utils._pytree.tree_leaves(args):
                    scale = get_scale(arg)
                    if scale is not None:
                        break

                unwrapped_args = torch.utils._pytree.tree_map(unwrap, args)
                unwrapped_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
                out = func(*unwrapped_args, **unwrapped_kwargs)

                def wrap(x):
                    # [FUNCTION DOC] HELPER: Wrap an inner-subclass result into the outer wrapper
                    # with the scale; other values pass through.
                    if isinstance(x, TensorWithCounter):
                        return NestedSubclass(x, scale)
                    return x

                return torch.utils._pytree.tree_map(wrap, out)

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x doubled plus one.
            return x * 2 + 1

        # Create nested structure: NestedSubclass -> TensorWithCounter -> tensors
        a = torch.rand(4, 4, requires_grad=True)
        b = torch.rand(4, 4, requires_grad=True)
        counter = Counter(start=3, end=10)
        size = SizeStore(4)
        inner = TensorWithCounter(a, b, counter, size)
        x = NestedSubclass(inner, scale=2.0)

        cnt = CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        out = opt_fn(x)

        # Output should preserve nested structure with same opaque objects
        self.assertTrue(isinstance(out, NestedSubclass))
        self.assertTrue(isinstance(out._inner, TensorWithCounter))
        self.assertIs(out._inner._counter, counter)
        self.assertIs(out._inner._size_store, size)
        self.assertEqual(out._scale, 2.0)

        # Test backward
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(isinstance(x.grad, NestedSubclass))

        self.assertEqual(cnt.frame_count, 1)

        # Call again with different instances that are __eq__ equal.
        a2 = torch.rand(4, 4, requires_grad=True)
        b2 = torch.rand(4, 4, requires_grad=True)
        counter2 = Counter(start=3, end=10)  # Same values, different instance
        size2 = SizeStore(4)  # Same value, different instance
        inner2 = TensorWithCounter(a2, b2, counter2, size2)
        x2 = NestedSubclass(inner2, scale=2.0)

        out2 = opt_fn(x2)
        self.assertEqual(cnt.frame_count, 1)  # No recompilation

        self.assertTrue(isinstance(out2, NestedSubclass))
        # Key checks: should use runtime instances, not traced instances
        self.assertIs(out2._inner._counter, counter2)
        # Value-type opaques in ctx are baked at compile time, so identity
        # is not preserved across re-invocations.
        self.assertEqual(out2._inner._size_store, size2)
        self.assertEqual(out2._scale, 2.0)

        out2.sum().backward()
        self.assertIsNotNone(x2.grad)
        self.assertTrue(isinstance(x2.grad, NestedSubclass))
        self.assertTrue(isinstance(x2.grad._inner, TensorWithCounter))
        self.assertIs(x2.grad._inner._counter, counter2)
        self.assertEqual(x2.grad._inner._size_store, size2)

    def test_value_type_opaque_in_tensor_attrs_errors(self):
        # [FUNCTION DOC] Checks that a wrapper subclass listing a value-type opaque among its
        # inner tensor attributes (it belongs in the context) fails to compile with a RuntimeError
        # saying only tensors and reference-type opaques are allowed there.
        """Value-type opaques must go in ctx, not tensor attrs."""

        class BadSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Deliberately wrong wrapper subclass that flattens a value-type
            # size store as a tensor attribute instead of as context.
            @staticmethod
            def __new__(cls, data, size_store):
                # [FUNCTION DOC] HELPER: Create the wrapper tensor with the data
                # tensor's metadata.
                kwargs = {
                    "strides": data.stride(),
                    "storage_offset": data.storage_offset(),
                    "device": data.device,
                    "layout": data.layout,
                    "requires_grad": data.requires_grad,
                    "dtype": data.dtype,
                }
                return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)

            def __init__(self, data, size_store):
                # [FUNCTION DOC] HELPER: Store the data tensor and size store.
                self._data = data
                self._size_store = size_store

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook that wrongly lists the size store as an
                # inner tensor attribute with empty context.
                # BUG: SizeStore is value-type, should be in ctx not tensor attrs
                return ["_data", "_size_store"], ()

            @staticmethod
            def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Unflatten hook: rebuild from the inner data and size
                # store mappings.
                return BadSubclass(inner_tensors["_data"], inner_tensors["_size_store"])

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch hook: unwraps subclass args, runs the op and re-
                # wraps tensor results with the same size store.
                if kwargs is None:
                    kwargs = {}

                def unwrap(x):
                    # [FUNCTION DOC] HELPER: Return the inner data for a subclass instance, else
                    # the value.
                    return x._data if isinstance(x, BadSubclass) else x

                size_store = None
                for arg in torch.utils._pytree.tree_leaves(args):
                    if isinstance(arg, BadSubclass):
                        size_store = arg._size_store
                        break

                unwrapped_args = torch.utils._pytree.tree_map(unwrap, args)
                unwrapped_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
                out = func(*unwrapped_args, **unwrapped_kwargs)

                def wrap(x):
                    # [FUNCTION DOC] HELPER: Wrap tensors into the bad subclass with the saved
                    # size store.
                    if isinstance(x, torch.Tensor):
                        return BadSubclass(x, size_store)
                    return x

                return torch.utils._pytree.tree_map(wrap, out)

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x doubled.
            return x * 2

        x = BadSubclass(torch.rand(4, 4), SizeStore(4))
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "SizeStore.*tensor attrs.*Only tensors and reference-type opaques",
        ):
            opt_fn(x)

    def test_multi_reference_opaque_identity(self):
        # [FUNCTION DOC] Registers a local reference-type Tag opaque and a subclass holding two
        # different reference opaques (counter and tag) as inner attributes. Checks outputs and
        # gradients keep each opaque by identity, including after a second call with equal-but-
        # distinct instances, with one compilation only.
        """Test identity preservation with multiple reference-type opaques in tensor attrs."""

        class Tag(CustomClassBase):
            # [CLASS DOC] HELPER: Local reference-type opaque holding a name with equality and
            # hash by name; registered symbolic inside the test.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Store the name.
                self.name = name

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if other is a Tag with the same name.
                return isinstance(other, Tag) and self.name == other.name

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash of the name.
                return hash(self.name)

        register_custom_class(Tag, typ="symbolic")

        class TwoRefSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass holding a data tensor plus an opaque
            # counter and an opaque tag as inner attributes.
            @staticmethod
            def __new__(cls, data, counter, tag):
                # [FUNCTION DOC] HELPER: Create the wrapper tensor with the data
                # tensor's metadata.
                kwargs = {
                    "strides": data.stride(),
                    "storage_offset": data.storage_offset(),
                    "device": data.device,
                    "layout": data.layout,
                    "requires_grad": data.requires_grad,
                    "dtype": data.dtype,
                }
                return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)

            def __init__(self, data, counter, tag):
                # [FUNCTION DOC] HELPER: Store the data tensor, counter and tag.
                self._data = data
                self._counter = counter
                self._tag = tag

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return a string showing the data tensor.
                return f"TwoRefSubclass({self._data})"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Flatten hook: names of data, counter and tag, with
                # empty context.
                return ["_data", "_counter", "_tag"], ()

            @staticmethod
            def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Unflatten hook: rebuild from the three inner attributes.
                return TwoRefSubclass(
                    inner_tensors["_data"],
                    inner_tensors["_counter"],
                    inner_tensors["_tag"],
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Dispatch hook: finds counter and tag from the first
                # subclass arg, unwraps, runs the op, and re-wraps tensor results with the same
                # counter and tag.
                if kwargs is None:
                    kwargs = {}

                def unwrap(x):
                    # [FUNCTION DOC] HELPER: Return the data tensor for a subclass instance, else
                    # the value.
                    return x._data if isinstance(x, TwoRefSubclass) else x

                counter = None
                tag = None
                for arg in torch.utils._pytree.tree_leaves(args):
                    if isinstance(arg, TwoRefSubclass):
                        counter = arg._counter
                        tag = arg._tag
                        break

                unwrapped_args = torch.utils._pytree.tree_map(unwrap, args)
                unwrapped_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
                out = func(*unwrapped_args, **unwrapped_kwargs)

                def wrap(x):
                    # [FUNCTION DOC] HELPER: Wrap tensors into the subclass with the found counter
                    # and tag.
                    if isinstance(x, torch.Tensor):
                        return TwoRefSubclass(x, counter, tag)
                    return x

                return torch.utils._pytree.tree_map(wrap, out)

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x doubled plus one.
            return x * 2 + 1

        counter = Counter(start=3, end=10)
        tag = Tag("test")
        x = TwoRefSubclass(torch.rand(4, 4, requires_grad=True), counter, tag)

        cnt = CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        out = opt_fn(x)

        self.assertIsInstance(out, TwoRefSubclass)
        self.assertIs(out._counter, counter)
        self.assertIs(out._tag, tag)

        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(cnt.frame_count, 1)

        # Second invocation with different-but-equal opaques
        counter2 = Counter(start=3, end=10)
        tag2 = Tag("test")
        self.assertEqual(counter, counter2)
        self.assertIsNot(counter, counter2)
        x2 = TwoRefSubclass(torch.rand(4, 4, requires_grad=True), counter2, tag2)

        out2 = opt_fn(x2)
        self.assertEqual(cnt.frame_count, 1)  # No recompilation

        self.assertIsInstance(out2, TwoRefSubclass)
        self.assertIs(out2._counter, counter2)
        self.assertIs(out2._tag, tag2)

        out2.sum().backward()
        self.assertIsNotNone(x2.grad)
        self.assertIsInstance(x2.grad, TwoRefSubclass)
        self.assertIs(x2.grad._counter, counter2)
        self.assertIs(x2.grad._tag, tag2)

    def test_opaque_obj_saved_for_backward(self):
        # [FUNCTION DOC] Checks an opaque multiplier object is saved for backward and passed to
        # the backward graph. Runs forward and backward with inductor and AOT eager recording
        # graphs and checks the dynamo graph and the forward and backward AOT graph texts, then,
        # with and without compiled autograd, checks gradients reflect the multiplier of each run
        # after a reset.
        """Test that opaque objects are correctly saved and passed to backward."""
        import torch._dynamo.compiled_autograd

        def foo(scale_obj, x):
            # [FUNCTION DOC] HELPER: Apply the scaling op to x and then double the result.
            result = torch.ops._TestOpaqueObject.mul_with_scale(scale_obj, x)
            result = result * 2
            return result

        def compile_and_run_with_backend(backend):
            # [FUNCTION DOC] HELPER: Compile the function with the given backend using a
            # multiplier of 2.5, check forward output and the gradient from an upstream gradient
            # of five.
            scale_obj = OpaqueMultiplier(2.5)
            x = torch.randn(3, 3, requires_grad=True)

            opt_f = torch.compile(foo, fullgraph=True, backend=backend)
            out = opt_f(scale_obj, x)
            expected = x * 2.5 * 2
            self.assertTrue(torch.allclose(out, expected))

            upstream_grad = torch.ones_like(out) * 5
            out.backward(upstream_grad)
            self.assertIsNotNone(x.grad)
            expected_grad = torch.ones_like(x) * 5 * 5
            self.assertTrue(torch.allclose(x.grad, expected_grad))

        backend = InductorAndRecordGraphs()
        compile_and_run_with_backend(backend)
        self.assertTrue(len(backend.graphs) > 0)
        fw_graph = backend.graphs[0]
        self.assertExpectedInline(
            fw_graph.code.strip(),
            f"""\
def forward(self, L_scale_obj_ : {_illegal_char_regex.sub("_", get_opaque_type_name(OpaqueMultiplier))}, L_x_ : torch.Tensor):
    l_scale_obj_ = L_scale_obj_
    l_x_ = L_x_
    mul_with_scale = torch.ops._TestOpaqueObject.mul_with_scale(l_scale_obj_, l_x_);  l_scale_obj_ = l_x_ = None
    mul = mul_with_scale * 2;  mul_with_scale = None
    return (mul,)""",
        )

        backend = AotEagerAndRecordGraphs()
        compile_and_run_with_backend(backend)
        self.assertTrue(len(backend.fw_graphs) > 0)
        fw_graph = backend.fw_graphs[0]
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul_with_scale = torch.ops._TestOpaqueObject.mul_with_scale.default(primals_1, primals_2);  primals_2 = None
    mul = torch.ops.aten.mul.Tensor(mul_with_scale, 2);  mul_with_scale = None
    return (mul, primals_1)""",
        )
        self.assertTrue(len(backend.bw_graphs) > 0)
        bw_graph = backend.bw_graphs[0]
        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, primals_1, tangents_1):
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
    get_multiplier_tensor = torch.ops._TestOpaqueObject.get_multiplier_tensor.default(primals_1, mul_1);  primals_1 = mul_1 = None
    return (None, get_multiplier_tensor)""",
        )

        for use_compiled_autograd in [False, True]:
            with self.subTest(use_compiled_autograd=use_compiled_autograd):
                torch._dynamo.reset()

                def run_with_multiplier(multiplier_value: float):
                    # [FUNCTION DOC] HELPER: Compile with AOT eager using the given
                    # multiplier value, run backward with or without compiled autograd as
                    # set by the enclosing loop, check output and gradient, and return a
                    # copy of the gradient.
                    scale_obj = OpaqueMultiplier(multiplier_value)
                    x = torch.randn(3, 3, requires_grad=True)

                    opt_f = torch.compile(foo, fullgraph=True, backend="aot_eager")
                    out = opt_f(scale_obj, x)
                    expected = x * multiplier_value * 2
                    self.assertTrue(torch.allclose(out, expected))

                    upstream_grad = torch.ones_like(out) * 5
                    with (
                        torch._dynamo.compiled_autograd._enable(
                            torch.compile(backend="eager")
                        )
                        if use_compiled_autograd
                        else contextlib.nullcontext()
                    ):
                        out.backward(upstream_grad)

                    self.assertIsNotNone(x.grad)
                    # gradient = upstream_grad * 2 * multiplier_value
                    expected_grad = torch.ones_like(x) * 5 * 2 * multiplier_value
                    self.assertTrue(torch.allclose(x.grad, expected_grad))
                    return x.grad.clone()

                grad1 = run_with_multiplier(2.5)
                torch._dynamo.reset()
                grad2 = run_with_multiplier(3.0)
                self.assertFalse(torch.allclose(grad1, grad2))

    def test_subgraph_tracer_create_arg_with_fake_script_object(self):
        # [FUNCTION DOC] Checks that accessing an attribute of an opaque class (a color constant)
        # inside a compiled function lifts the opaque instance as a graph input with a global-
        # named placeholder, that the opaque class and its instance type are symbolic opaque
        # types, and that the numeric result and graph text are as expected. Prints the captured
        # graph code (side effect).
        """Test that opaque class attribute access works correctly.

        This tests the code path where:
        1. An opaque class (like Color) is accessed via CustomClassVariable
        2. Attribute access (Color.RED) goes through tp_getattro_impl with static getattr
        3. The opaque object is correctly lifted as a graph input
        """
        from torch._library.opaque_object import is_opaque_symbolic_type

        self.assertTrue(is_opaque_symbolic_type(Color))
        self.assertTrue(is_opaque_symbolic_type(type(Color.RED)))

        captured = {"graph": None, "example_inputs": None}

        def capture_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Custom backend that records the graph and example inputs in
            # the outer dictionary and returns the graph module unchanged.
            captured["graph"] = gm
            captured["example_inputs"] = example_inputs
            return gm

        @torch.compile(fullgraph=True, backend=capture_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply the color scale op using the GREEN color constant.
            return torch.ops._TestOpaqueObject.apply_color_scale(Color.GREEN, x)

        x = torch.randn(3, 3)
        result = fn(x)

        self.assertIsNotNone(captured["graph"])
        print(captured["graph"].code.strip())

        expected = x * float(Color.GREEN.value)
        self.assertTrue(torch.allclose(result, expected))

        graph_code = captured["graph"].code.strip()
        self.assertExpectedInline(
            graph_code,
            f"""\
def forward(self, G_Color_GREEN : {_illegal_char_regex.sub("_", get_opaque_type_name(Color))}, L_x_ : torch.Tensor):
    g_color_green = G_Color_GREEN
    l_x_ = L_x_
    apply_color_scale = torch.ops._TestOpaqueObject.apply_color_scale(g_color_green, l_x_);  g_color_green = l_x_ = None
    return (apply_color_scale,)""",
        )

    def test_hoist_basic(self):
        # [FUNCTION DOC] Checks a hoisted string opaque is lifted out of the graph: with the eager
        # and the AOT eager recording backends the recorded graph text must not contain the string
        # value, and with inductor the result must equal doubling.
        def f(x):
            # [FUNCTION DOC] HELPER: Apply the string op to x and a hoisted string saying double.
            return op_with_string(x, HoistedString("double"))

        x = torch.tensor(3.0)

        # backend = "eager"
        backend = EagerAndRecordGraphs()
        _ = torch.compile(f, fullgraph=True, backend=backend)(x)
        graph = backend.graphs[0]
        self.assertNotIn("double", str(graph))

        # backend = "aot_eager"
        backend = AotEagerAndRecordGraphs()
        _ = torch.compile(f, fullgraph=True, backend=backend)(x)
        graph = backend.graphs[0]
        self.assertNotIn("double", str(graph))

        # backend = "inductor"
        y = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(y, 2 * x)

    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
        }
    )
    def test_hoist_cache_hits(self):
        # [FUNCTION DOC] Run with autograd cache and FX graph cache enabled (remote cache off).
        # Under a fresh inductor cache, compiles two functions differing only in the hoisted
        # string value and checks the counters: first function misses and saves in both caches;
        # the second hits both caches.
        torch._dynamo.reset()
        counters.clear()

        with fresh_inductor_cache():
            # Because HoistedString should not be in the graph, the following
            # two functions should share AOTAutogradCache and FXGraphCache entries

            @torch.compile(fullgraph=True)
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function applying the string op with the
                # double string.
                return op_with_string(x, HoistedString("double"))

            @torch.compile(fullgraph=True)
            def g(x):
                # [FUNCTION DOC] HELPER: Compiled function applying the string op with the
                # square string.
                return op_with_string(x, HoistedString("square"))

            x = torch.tensor(3.0)

            f(x)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

            g(x)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

    def test_hoisted_value_type_make_fx(self):
        # [FUNCTION DOC] Trace with make_fx a function applying the string op to a hoisted string
        # input; checks the graph keeps the string as an input and that the traced module gives
        # doubled and squared results when called with the corresponding strings.
        def foo(x, hoisted_str):
            # [FUNCTION DOC] HELPER: Apply the string op to x and the given hoisted string.
            return op_with_string(x, hoisted_str)

        x = torch.randn(3, 3)
        gm = make_fx(foo)(x, HoistedString("double"))

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1, hoisted_str_1):
    op_with_string = torch.ops.mylib.op_with_string.default(x_1, hoisted_str_1);  x_1 = hoisted_str_1 = None
    return op_with_string""",
        )
        self.assertEqual(gm(x, HoistedString("double")), x * 2)
        self.assertEqual(gm(x, HoistedString("square")), x * x)

    def test_opaque_class_literal_attribute_inlined(self):
        # [FUNCTION DOC] Checks a literal float class attribute of an opaque class used in a
        # compiled function is folded as a constant in the graph (no graph input), and the
        # result matches.
        """Test that literal attributes on opaque classes are inlined without source tracking.

        When accessing a literal class attribute (like Color.DEFAULT_SCALE = 1.5),
        the value should be constant-folded directly without creating a graph input.
        """
        captured = {"graph": None}

        def capture_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend recording the graph module in the outer dictionary
            # and returning it.
            captured["graph"] = gm
            return gm

        @torch.compile(fullgraph=True, backend=capture_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiply x by the class-level default scale.
            return x * Color.DEFAULT_SCALE

        x = torch.randn(3, 3)
        result = fn(x)

        expected = x * 1.5
        self.assertTrue(torch.allclose(result, expected))

        self.assertIsNotNone(captured["graph"])
        graph_code = captured["graph"].code.strip()
        self.assertExpectedInline(
            graph_code,
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    mul = l_x_ * 1.5;  l_x_ = None
    return (mul,)""",
        )

    def test_opaque_class_missing_attribute_graph_break(self):
        # [FUNCTION DOC] Checks that using a nonexistent attribute of an opaque class in a full-
        # graph compile raises the dynamo Unsupported error saying the attribute was not found.
        """Test that accessing a non-existent attribute on an opaque class causes a graph break.

        This tests GB7685: "Attribute not found on opaque class"
        """

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiply x by a missing class attribute.
            return x * Color.NONEXISTENT

        x = torch.randn(3, 3)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Attribute not found on opaque class",
        ):
            fn(x)

    def test_opaque_class_unsupported_descriptor_graph_break(self):
        # [FUNCTION DOC] Checks that using a class attribute with an unsupported descriptor in a
        # full-graph compile raises the dynamo Unsupported error about unsupported descriptors.
        """Test that accessing an unsupported descriptor on an opaque class causes a graph break.

        This tests GB9567: "Unsupported descriptor on opaque class"
        """

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiply x by the descriptor attribute.
            return x * ColorWithDescriptor.custom_prop

        x = torch.randn(3, 3)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Unsupported descriptor on opaque class",
        ):
            fn(x)

    def test_opaque_class_staticmethod(self):
        # [FUNCTION DOC] Checks that reading a static method attribute off an opaque class during
        # compilation is supported: result equals doubling and a graph was captured.
        """Test that accessing a staticmethod on an opaque class works correctly.

        This verifies that CustomClassVariable.tp_getattro_impl properly handles
        staticmethod descriptors (instead of raising 'Unsupported descriptor').
        """
        captured = {"graph": None}

        def capture_backend(gm, _):
            # [FUNCTION DOC] HELPER: Backend recording the graph module and returning it.
            captured["graph"] = gm
            return gm

        @torch.compile(fullgraph=True, backend=capture_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Read the static method attribute from the class (discarded)
            # and return x doubled.
            _ = Color.RED_STATIC
            return x * 2

        x = torch.randn(3, 3)
        result = fn(x)
        expected = x * 2
        self.assertTrue(torch.allclose(result, expected))
        self.assertIsNotNone(captured["graph"])

    def test_opaque_class_property(self):
        # [FUNCTION DOC] Checks that reading a property attribute off an opaque class (yielding
        # the property object) during compilation is supported: result equals doubling and a graph
        # was captured.
        """Test that accessing a property descriptor on an opaque class works correctly.

        This verifies that CustomClassVariable.tp_getattro_impl properly handles
        property descriptors. When accessing a property on the class (not instance),
        you get the property object back.
        """
        captured = {"graph": None}

        def capture_backend(gm, _):
            # [FUNCTION DOC] HELPER: Backend recording the graph module and returning it.
            captured["graph"] = gm
            return gm

        @torch.compile(fullgraph=True, backend=capture_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Read the property attribute from the class (discarded) and
            # return x doubled.
            _ = Color.name
            return x * 2

        x = torch.randn(3, 3)
        result = fn(x)
        expected = x * 2
        self.assertTrue(torch.allclose(result, expected))
        self.assertIsNotNone(captured["graph"])

    def test_multiple_opaque_objects_in_backward(self):
        # [FUNCTION DOC] Run the multi-RNG op eagerly and then compiled, with backward each time,
        # checking the gradient is 1.5 times the upstream gradient; guards against placing saved
        # opaque objects at the wrong position when calling backward.
        """Test that multiple opaque objects are correctly positioned in backward pass.

        This test verifies that when setup_context saves multiple opaque objects,
        AOTAutograd correctly positions them when calling the backward function.
        This addresses a bug where opaque objects were placed in the wrong position.
        """
        # Test eager mode
        x = torch.randn(1, requires_grad=True)
        rng1 = RNGState(123)
        rng2 = RNGState(456)

        result = torch.ops._TestOpaqueObject.multi_rng(x, rng1, rng2)
        grad_o = torch.rand_like(result)
        result.backward(grad_o)
        self.assertEqual(x.grad, grad_o * 1.5)
        x.grad = None

        # Test compiled mode - this should not crash with incorrect opaque object positioning
        x2 = torch.randn(1, requires_grad=True)
        compiled_fn = torch.compile(
            torch.ops._TestOpaqueObject.multi_rng,
            fullgraph=True,
        )
        result = compiled_fn(x2, rng1, rng2)
        grad_o = torch.rand_like(result)
        result.backward(grad_o)
        self.assertEqual(x2.grad, grad_o * 1.5)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_invoke_subgraph(self):
        # [FUNCTION DOC] Run with single-use subgraph inlining off. Checks that a function using a
        # nested compile region over an opaque multiplier produces an invoke-subgraph call taking
        # the opaque as an input, with the expected readable graph (type name normalized), and
        # that output and gradients match eager.
        @torch.compiler.nested_compile_region
        def fn(scale_obj, x):
            # [FUNCTION DOC] HELPER: Nested compile region: scale x with the op then double.
            result = torch.ops._TestOpaqueObject.mul_with_scale(scale_obj, x)
            result = result * 2
            return result

        def gn(scale_obj, x):
            # [FUNCTION DOC] HELPER: Call the region and return the doubled sum of its output
            # with itself.
            z = fn(scale_obj, x)
            return z + z

        scale_obj = OpaqueMultiplier(3.0)
        x = torch.randn(2, 2, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)

        ref = gn(scale_obj, x_clone)
        ref.sum().backward()

        backend = AotEagerAndRecordGraphs()
        opt_outer = torch.compile(gn, fullgraph=True, backend=backend)
        res = opt_outer(scale_obj, x)
        res.sum().backward()

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        # Normalize module-qualified opaque type names since they differ
        # depending on how the test is invoked (__main__ vs test_opaque_obj_v2).
        actual = re.sub(r"\w+_OpaqueMultiplier", "OpaqueMultiplier", actual)
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_scale_obj_ : OpaqueMultiplier, L_x_: "f32[2, 2]"):
        l_scale_obj_ = L_scale_obj_
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_scale_obj_, l_x_);  subgraph_0 = l_scale_obj_ = l_x_ = None
        getitem: "f32[2, 2]" = invoke_subgraph[0];  invoke_subgraph = None

        add: "f32[2, 2]" = getitem + getitem;  getitem = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_scale_obj_ : OpaqueMultiplier, l_x_: "f32[2, 2]"):
            result: "f32[2, 2]" = torch.ops._TestOpaqueObject.mul_with_scale(l_scale_obj_, l_x_);  l_scale_obj_ = l_x_ = None

            result_1: "f32[2, 2]" = result * 2;  result = None
            return (result_1,)
""",
        )

        self.assertEqual(ref, res)
        self.assertEqual(ref.grad, res.grad)

    def test_value_type_subclass(self):
        # [FUNCTION DOC] Checks a subclass of a registered constant opaque type is itself
        # recognized as opaque and can be used in a compiled function, with output equal to eager.
        """Subclasses of registered opaque types should be recognized as opaque."""

        class ExtendedConfig(ValueConfig):
            # [CLASS DOC] HELPER: Constant-type config subclass adding a scale value; equality,
            # hash and FX representation include both mode and scale.
            def __init__(self, mode: str, scale: float):
                # [FUNCTION DOC] HELPER: Initialize the parent with the mode and store the scale.
                super().__init__(mode)
                self.scale = scale

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if other is an extended config with the same mode
                # and scale.
                return (
                    isinstance(other, ExtendedConfig)
                    and self.mode == other.mode
                    and self.scale == other.scale
                )

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash of the pair of mode and scale.
                return hash((self.mode, self.scale))

            def __fx_repr__(self):
                # [FUNCTION DOC] HELPER: Return the pair of constructor text with mode and scale
                # and the class mapping.
                return (
                    f"ExtendedConfig(mode={self.mode!r}, scale={self.scale!r})",
                    {"ExtendedConfig": ExtendedConfig},
                )

        self.assertTrue(is_custom_class(ExtendedConfig))
        self.assertTrue(is_opaque_constant_type(ExtendedConfig))

        cfg = ExtendedConfig("square", 2.0)

        def foo(x):
            # [FUNCTION DOC] HELPER: Apply the config op to x with the captured extended config.
            return torch.ops._TestOpaqueObject.process_with_config(x, cfg)

        x = torch.randn(3, 3)
        opt_f = torch.compile(foo, fullgraph=True, backend="aot_eager")
        res = opt_f(x)
        self.assertEqual(res, foo(x))

    def test_opaque_object_with_inductor_backend(self):
        # [FUNCTION DOC] Symbolically trace a module holding an opaque color as an attribute
        # (producing an attribute fetch node), compile the graph with the inductor compile entry
        # point, and check the result equals scaling by the color value.
        """Test that opaque objects work correctly with inductor's get_attr handling."""

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an opaque color attribute, whose forward applies the
            # color-scale op.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the parent module state and store the RED
                # color attribute.
                super().__init__()
                self.color = Color.RED

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scale x with the color-scale op using the stored color.
                return torch.ops._TestOpaqueObject.apply_color_scale(self.color, x)

        mod = TestModule()
        x = torch.randn(3, 3)

        gm = torch.fx.symbolic_trace(mod)

        has_get_attr = any(node.op == "get_attr" for node in gm.graph.nodes)
        self.assertTrue(has_get_attr, "expected get_attr node for opaque object")

        compiled_fn = compile_fx(gm, [x])
        result = compiled_fn(x)

        expected = x * float(Color.RED.value)
        self.assertTrue(torch.allclose(result, expected))

    def test_captured_opaque_object_export(self):
        # [FUNCTION DOC] Exports (strict) a module whose forward closes over a module-level RNG
        # object captured via a closure, and checks the exported graph lifts the object as a
        # custom-object input.
        """Test that opaque objects captured from closures work with torch.export.

        When an opaque object is captured via a module hook closure, it gets a
        CellContentsSource instead of GetItemSource. The export path needs to
        handle this gracefully.
        """
        rng = RNGState(42)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose forward calls the noisy-inject
            # op with the closed-over RNG (result unused) and then applies the linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the parent module state and the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Call noisy inject (discarding the result) and return the
                # linear output.
                torch.ops._TestOpaqueObject.noisy_inject(x, rng)
                return self.linear(x)

        ep = torch.export.export(Model(), (torch.randn(4, 4),), strict=True)
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, obj_lifted_custom_0, x):
    noisy_inject_default = torch.ops._TestOpaqueObject.noisy_inject.default(x, obj_lifted_custom_0);  obj_lifted_custom_0 = noisy_inject_default = None
    linear_default = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    return (linear_default,)""",
        )

    def test_hoist_no_recompile_on_different_string(self):
        # [FUNCTION DOC] Checks that calling a compiled function with different hoisted-string
        # values does not recompile: two calls with different strings give correct results and
        # frame count stays one.
        cnt = CompileCounter()

        def f(x, label):
            # [FUNCTION DOC] HELPER: Apply the string op to x with a hoisted string built from
            # the label.
            return op_with_string(x, HoistedString(label))

        opt_f = torch.compile(f, backend=cnt, fullgraph=True)
        x = torch.tensor(3.0)

        res1 = opt_f(x, "double")
        self.assertEqual(res1, f(x, "double"))
        self.assertEqual(cnt.frame_count, 1)

        res2 = opt_f(x, "square")
        self.assertEqual(res2, f(x, "square"))
        self.assertEqual(cnt.frame_count, 1)

    def test_opaque_multi_output_not_tensor_irnode(self):
        # [FUNCTION DOC] Checks the inductor opaque multi-output node object is an IR node but is
        # not classified as a tensor IR node, nor is a mocked non-tensor object or an integer.
        """OpaqueMultiOutput must not be classified as a tensor IR node.

        _is_tensor_irnode is used to select nodes for stride constraint
        functions (require_stride1, require_contiguous, etc). If
        OpaqueMultiOutput passes the check, those functions would call
        get_stride/get_dtype/make_loader on it and crash."""
        from unittest.mock import MagicMock

        from torch._inductor import ir
        from torch._inductor.lowering import _is_tensor_irnode

        opaque_multi = ir.OpaqueMultiOutput.__new__(ir.OpaqueMultiOutput)
        self.assertIsInstance(opaque_multi, ir.IRNode)
        self.assertFalse(_is_tensor_irnode(opaque_multi))

        non_tensor = MagicMock(spec=ir.NonTensorObj)
        non_tensor.__class__ = ir.NonTensorObj
        self.assertFalse(_is_tensor_irnode(non_tensor))

        self.assertFalse(_is_tensor_irnode(42))

    @unittest.skipIf(not dist.is_available(), "requires distributed")
    def test_fake_script_object_process_group_pybind(self):
        # [FUNCTION DOC] Skipped without distributed. Tests that a fake wrapper of a process group
        # flows from one distributed op to another under compile. It tears down any existing
        # process group, starts a fake-backend group with two ranks, registers distributed opaque
        # types, builds a device mesh, compiles a function using the mesh process group size,
        # compares to eager, and restores the initial process group state in a finally block.
        """FakeScriptObject wrapping ProcessGroup must be unwrapped in the
        pybind toIValue before casting to c10::intrusive_ptr<ProcessGroup>.

        During Dynamo tracing with CooR, mesh_get_process_group returns a
        FakeScriptObject-wrapped ProcessGroup. When that flows into a C++
        custom op, toIValue needs to unwrap real_obj before the pybind
        cast. This test exercises that path by calling
        mesh_get_process_group (which returns a wrapped PG during tracing)
        and passing it to another op that consumes the ProcessGroup."""
        from torch.distributed.device_mesh import (
            _register_distributed_opaque_types,
            DeviceMesh,
        )
        from torch.testing._internal.distributed.fake_pg import FakeStore

        already_initialized = dist.is_initialized()
        if already_initialized:
            dist.destroy_process_group()

        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        try:
            _register_distributed_opaque_types()
            mesh = DeviceMesh("cpu", torch.arange(2))

            def f(mesh, x):
                # [FUNCTION DOC] HELPER: Get the mesh process group for dimension zero and add its
                # size to x.
                pg = torch.ops._dtensor.mesh_get_process_group(mesh, 0)
                return x + pg.size()

            x = torch.randn(4)
            compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
            result = compiled_f(mesh, x)
            expected = f(mesh, x)
            self.assertEqual(result, expected)
        finally:
            dist.destroy_process_group()
            if already_initialized:
                dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)

    def test_enum_export(self):
        # [FUNCTION DOC] Exports (non-strict) a module that adds an enum member's value, then
        # checks the exported module gives the right result and the graph folds the value as a
        # constant. Then compiles the module with the eager recording backend and checks the enum
        # value is folded in the dynamo graph.
        class Direction(enum.Enum):
            # [CLASS DOC] HELPER: Local enum with two members, UP and DOWN, with integer values
            # zero and one.
            UP = 0
            DOWN = 1

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds the integer value of the passed enum
            # member to x.
            def forward(self, x, d):
                # [FUNCTION DOC] HELPER: Return x plus the enum member's value.
                return x + d.value

        ep = torch.export.export(Mod(), (torch.randn(4, 4), Direction.UP), strict=False)
        self.assertEqual(
            ep.module()(torch.ones(4, 4), Direction.UP),
            torch.ones(4, 4) + Direction.UP.value,
        )
        self.assertExpectedInline(
            normalize_gm(ep.graph_module.print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[4, 4]", d):
        add_tensor: "f32[4, 4]" = torch.ops.aten.add.Tensor(x, 0);  x = None
        return (add_tensor,)
""",
        )

        backend = EagerAndRecordGraphs()
        opt_fn = torch.compile(Mod(), backend=backend)
        x = torch.randn(4, 4)
        res = opt_fn(x, Direction.UP)
        self.assertEqual(
            res,
            x + Direction.UP.value,
        )
        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 4]"):
        l_x_ = L_x_

        add: "f32[4, 4]" = l_x_ + 0;  l_x_ = None
        return (add,)
""",
        )

    def test_enum_custom_op(self):
        # [FUNCTION DOC] Defines a custom op taking an enum-typed argument, compiles a function
        # calling it with a color member under full-graph mode, and checks the result matches
        # eager and the dynamo graph passes the enum member directly to the op.
        def get_color():
            # [FUNCTION DOC] HELPER: Define and return a local enum class with RED, GREEN and
            # BLUE members.
            class Color(enum.Enum):
                # [CLASS DOC] HELPER: Local enum with RED, GREEN and BLUE integer members.
                RED = 0
                GREEN = 1
                BLUE = 2

            return Color

        Color = get_color()

        @torch.library.custom_op("test_enum::add_color", mutates_args=())
        def add_color(x: torch.Tensor, c: Color) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op (no mutation) adding the enum member's value to
            # the tensor.
            return x + c.value

        @add_color.register_fake
        def _(x, c):
            # [FUNCTION DOC] HELPER: Fake implementation of the add_color op; empty tensor like x.
            return torch.empty_like(x)

        def fn(x, c):
            # [FUNCTION DOC] HELPER: Call the add_color op with x and the color.
            return add_color(x, c)

        x = torch.randn(4, 4)
        ref = fn(x, Color.GREEN)
        backend = EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        res = opt_fn(x, Color.GREEN)
        self.assertEqual(ref, res)
        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 4]"):
        l_x_ = L_x_

        add_color_default: "f32[4, 4]" = torch.ops.test_enum.add_color.default(l_x_, Color.GREEN);  l_x_ = None
        return (add_color_default,)
""",
        )

    def test_subclass_parametrization_with_opaque_attrs(self):
        # [FUNCTION DOC] Checks unwrapping of subclass parameters in a module when the subclass
        # carries opaque attributes: builds a module with a parameter that is a TensorWithCounter,
        # runs the unwrap function, and runs the module once; must not raise.
        """unwrap_tensor_subclass_parameters should handle non-tensor attrs."""
        from torch._functorch._aot_autograd.subclass_parametrization import (
            unwrap_tensor_subclass_parameters,
        )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter wrapping a TensorWithCounter and a
            # forward adding it to x.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create two random tensors and wrap them with a counter
                # and size store as a parameter.
                super().__init__()
                a = torch.randn(3, 4)
                b = torch.randn(3, 4)
                self.p = torch.nn.Parameter(
                    TensorWithCounter(a, b, Counter(1, 10), SizeStore(4))
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x plus the parameter.
                return x + self.p

        m = M()
        unwrap_tensor_subclass_parameters(m)
        x = torch.randn(3, 4)
        m(x)

    def test_export_non_strict_with_opaque_attrs(self):
        # [FUNCTION DOC] Checks non-strict export accepts a TensorWithCounter input (opaque
        # attributes present) without raising.
        """torch.export with strict=False should handle subclasses with non-tensor attrs."""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward doubles the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x doubled.
                return x * 2

        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        x = TensorWithCounter(a, b, Counter(1, 10), SizeStore(4))
        torch.export.export(M(), (x,), strict=False)

    @unittest.skipIf(not dist.is_available(), "requires distributed")
    def test_get_untyped_storages_with_opaque_attrs(self):
        # [FUNCTION DOC] Skipped without distributed. Checks the distributed tool helper that
        # collects untyped storages from a tensor handles a TensorWithCounter: under warnings
        # turned into errors, it must not warn and must return a non-empty collection.
        """get_untyped_storages should skip non-tensor attrs without warning."""
        from torch.distributed._tools.common_utils import get_untyped_storages

        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        x = TensorWithCounter(a, b, Counter(1, 10), SizeStore(4))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            storages = get_untyped_storages(x)
        # Should get storages from the two inner tensors (a and b)
        self.assertTrue(len(storages) > 0)

    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
        }
    )
    def test_subclass_opaque_attrs_cache_hit(self):
        # [FUNCTION DOC] Run with autograd cache and FX graph cache enabled (remote off). Under a
        # fresh inductor cache, compiles a doubling function on a TensorWithCounter, checks
        # identity of output opaques and one cache miss; after a dynamo reset recompiles an
        # equivalent function on new equal opaque instances and checks one cache hit and that
        # runtime opaque instances (not the serialized ones) appear in the output.
        """opaque_attrs in SubclassCreationMeta should survive AOTAutograd cache round-trip."""
        torch._dynamo.reset()
        counters.clear()

        with fresh_inductor_cache():

            @torch.compile(fullgraph=True)
            def fn(x):
                # [FUNCTION DOC] HELPER: Return x doubled.
                return x * 2

            a = torch.randn(4, 4)
            b = torch.randn(4, 4)
            counter = Counter(start=3, end=10)
            size = SizeStore(4)
            x = TensorWithCounter(a, b, counter, size)

            out = fn(x)
            self.assertIsInstance(out, TensorWithCounter)
            self.assertIs(out._counter, counter)
            self.assertIs(out._size_store, size)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            # Reset dynamo to force a fresh compilation that should hit the cache
            torch._dynamo.reset()

            @torch.compile(fullgraph=True)
            def fn2(x):
                # [FUNCTION DOC] HELPER: Return x doubled (second function used to force a
                # fresh compile).
                return x * 2

            a2 = torch.randn(4, 4)
            b2 = torch.randn(4, 4)
            counter2 = Counter(start=3, end=10)
            size2 = SizeStore(4)
            x2 = TensorWithCounter(a2, b2, counter2, size2)

            out2 = fn2(x2)
            self.assertIsInstance(out2, TensorWithCounter)
            # Runtime opaque values should be used, not the serialized ones
            self.assertIs(out2._counter, counter2)
            self.assertEqual(out2._size_store, size2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    def test_op_passthrough_counter_in_tuple(self):
        # [FUNCTION DOC] Defines an op that takes a counter and tensor and returns the counter
        # itself together with a tensor, whose fake also passes the counter through. Compiles
        # a function that uses the returned counter in another op, and checks the result
        # matches eager.
        # When a fake kernel returns its Counter input directly, the getitem
        # proxy's example_value is already a FakeScriptObject.
        counter_type = get_opaque_type_name(Counter)
        torch.library.define(
            "_TestOpaqueObject::passthrough_counter",
            f"({counter_type} c, Tensor x) -> ({counter_type}, Tensor)",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::passthrough_counter",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def passthrough_impl(c: Counter, x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Real implementation: return the same counter and the tensor
            # times the counter start.
            return c, x * c.start

        @torch.library.register_fake(
            "_TestOpaqueObject::passthrough_counter", lib=self.lib
        )
        def passthrough_fake(c: Counter, x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Fake implementation: return the same counter and an empty
            # tensor like x.
            return c, torch.empty_like(x)

        def fn(c, x):
            # [FUNCTION DOC] HELPER: Pass the counter through the op, then add the start tensor of
            # the returned counter to the returned tensor.
            out_c, out_x = torch.ops._TestOpaqueObject.passthrough_counter(c, x)
            return torch.ops._TestOpaqueObject.counter_start(out_c) + out_x

        c = Counter(3, 10)
        x = torch.randn(4)
        ref = fn(c, x)
        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        res = opt_fn(c, x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_script_object_intermediate_exposed_from_checkpoint(self):
        # [FUNCTION DOC] Run with forward side effects in backward under checkpoint skipped.
        # Checks an opaque counter created inside an activation-checkpoint region and stored
        # through a list side effect is exposed as a subgraph output, by compiling with the AOT
        # eager decomposition partition backend and comparing with eager.
        # A CustomClassObjectVariable created inside an AC region and accessed
        # outside via a list side effect must be exposed as a subgraph output.
        import torch.utils.checkpoint

        def gn(x, results):
            # [FUNCTION DOC] HELPER: Checkpointed region: create a counter via an op, append it to
            # the given list (side effect), and return x doubled.
            counter = torch.ops._TestOpaqueObject.create_counter(x.shape[0], x.shape[0])
            results.append(counter)
            return x * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Run the region under non-reentrant checkpoint and add the
            # start tensor of the counter that was appended in the list to the output.
            results = []
            out = torch.utils.checkpoint.checkpoint(gn, x, results, use_reentrant=False)
            return torch.ops._TestOpaqueObject.counter_start(results[0]) + out

        x = torch.randn(3, 4)
        ref = fn(x)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager_decomp_partition")
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_reference_type_opaque_object_state(self):
        # [FUNCTION DOC] Builds an FX graph by hand whose placeholder metadata value is a raw
        # opaque reference object (not wrapped in a fake), mimicking compile-on-one-rank, with a
        # call to the scaling op, compiles it with the inductor inner compile entry and checks the
        # result is x times two.
        """When compile_fx_inner receives an FX graph whose placeholder meta['val']
        is a raw opaque reference type (not wrapped in FakeScriptObject), inductor
        handles it via OpaqueObjectState. This codepath is used by compile-on-one-rank
        (COOR) where the FX graph is constructed with real objects."""
        m = OpaqueMultiplier(2.0)
        x = torch.ones(3)

        graph = torch.fx.Graph()
        m_node = graph.placeholder("m")
        m_node.meta["val"] = m
        fake_mode = FakeTensorMode()
        x_node = graph.placeholder("x")
        x_node.meta["val"] = fake_mode.from_tensor(x)
        out = graph.call_function(
            torch.ops._TestOpaqueObject.mul_with_scale.default, (m_node, x_node)
        )
        out.meta["val"] = fake_mode.from_tensor(x)
        graph.output((out,))

        gm = torch.fx.GraphModule({}, graph)
        compiled = compile_fx_inner(gm, [m, x])
        result = compiled([m, x])
        self.assertEqual(result, (x * 2,))

    @unittest.skipIf(
        not has_cpp_wrapper_for_device("cpu"),
        "requires CPU cpp wrapper",
    )
    @inductor_config.patch(cpp_wrapper=True)
    def test_cpp_wrapper_opaque_object_state_input_slot(self):
        # [FUNCTION DOC] Skipped without a CPU C++ wrapper; runs with the C++ wrapper enabled.
        # Builds an FX graph of tensor add that also has an unused raw opaque placeholder between
        # two tensor inputs, compiles it with the C++ wrapper, and checks the result equals x plus
        # y (the opaque input slot must not shift the other inputs).
        m = OpaqueMultiplier(2.0)
        x = torch.ones(3)
        y = torch.arange(3, dtype=torch.float32)

        graph = torch.fx.Graph()
        fake_mode = FakeTensorMode()
        x_node = graph.placeholder("x")
        x_node.meta["val"] = fake_mode.from_tensor(x)
        m_node = graph.placeholder("m")
        m_node.meta["val"] = m
        y_node = graph.placeholder("y")
        y_node.meta["val"] = fake_mode.from_tensor(y)
        out = graph.call_function(torch.ops.aten.add.Tensor, (x_node, y_node))
        out.meta["val"] = fake_mode.from_tensor(x + y)
        graph.output((out,))

        gm = torch.fx.GraphModule({}, graph)
        compiled = compile_fx_inner(gm, [x, m, y], cpp_wrapper=True)
        result = compiled([x, m, y])
        self.assertEqual(result, (x + y,))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_benchmark_harness_no_pickle_for_opaque_inputs(self):
        # [FUNCTION DOC] Skipped without GPU. Compiles a simple function on a GPU
        # TensorWithCounter with inductor, collects generated code, and checks that there is code
        # and that none of it contains the pickle module name, so opaque inputs are not pickled in
        # the benchmark harness.
        """Opaque graph inputs must not be pickled in the benchmark harness."""
        a = torch.randn(4, 4, device=GPU_TYPE)
        b = torch.randn(4, 4, device=GPU_TYPE)
        twc = TensorWithCounter(a, b, Counter(0, 10), SizeStore(4))

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus one.
            return x + 1

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        _, codes = run_and_get_code(compiled_fn, twc)
        self.assertGreater(len(codes), 0)
        for code in codes:
            self.assertNotIn("pickle", code)

    def test_opaque_object_state_in_graph_output(self):
        # [FUNCTION DOC] Builds an FX graph by hand with a raw opaque multiplier placeholder where
        # the graph outputs include both the op result and the opaque object itself (as saved-for-
        # backward objects appear after partitioning). Compiles with the inductor inner compile
        # entry, which must accept opaque state among outputs and in memory planning. Checks the
        # first output equals x times two and the second output is the identical opaque object.
        """When compile_fx_inner receives a graph where a raw opaque reference
        type appears in the outputs (not just inputs), inductor must handle it.
        This happens in CooR (compile-on-one-rank) precompile: aot_autograd
        partitions the joint graph so the forward graph saves opaque objects
        (e.g. DeviceMesh) for the backward graph by including them in forward
        outputs.  Inductor must:
          1. Accept OpaqueObjectState in the output type assertion
             (GraphLowering.output).
          2. Handle NonTensorObj graph inputs in the memory planner
             (get_dep_size_hint) without calling get_numel/get_dtype on them."""
        m = OpaqueMultiplier(2.0)
        x = torch.ones(3)

        graph = torch.fx.Graph()
        m_node = graph.placeholder("m")
        m_node.meta["val"] = m
        fake_mode = FakeTensorMode()
        x_node = graph.placeholder("x")
        x_node.meta["val"] = fake_mode.from_tensor(x)
        out = graph.call_function(
            torch.ops._TestOpaqueObject.mul_with_scale.default, (m_node, x_node)
        )
        out.meta["val"] = fake_mode.from_tensor(x)
        # Include the opaque object in the output tuple, simulating how
        # aot_autograd's forward graph passes saved-for-backward objects
        # through as outputs.
        graph.output((out, m_node))

        gm = torch.fx.GraphModule({}, graph)
        compiled = compile_fx_inner(gm, [m, x])
        result = compiled([m, x])
        self.assertEqual(result[0], x * 2)
        self.assertIs(result[1], m)

    def test_reconstruct_fn_sets_meta_val(self):
        # [FUNCTION DOC] Regression test: a node created by the reconstruct hook of an opaque type
        # must have value metadata and be classified as opaque. Defines an op deriving a
        # multiplier from a parent, temporarily installs a reconstruct hook on the registry entry
        # of the multiplier type, traces with make_fx a function capturing a derived child from a
        # closure, finds derive nodes in the graph and asserts each has value metadata and is
        # classified opaque. Restores the original hook in a finally block.
        """Opaque nodes created via reconstruct_fn have meta['val'] set.

        When _try_reconstruct_opaque dispatches through a custom op via
        Proxy.__torch_function__, the resulting node does not get
        meta['val'] set automatically (unlike the __torch_dispatch__
        path which calls track_tensor_tree).  The fix in
        _try_reconstruct_opaque ensures set_meta is called so that
        downstream consumers like the min-cut partitioner can classify
        the node correctly via is_opaque_node().
        """
        from torch._functorch._aot_autograd.graph_compile import is_opaque_node

        # Use the already-registered OpaqueMultiplier type.
        # Register a reconstruct_fn that derives one multiplier from
        # another via a custom op — mirrors how DeviceMesh submeshes
        # are derived from a parent mesh via _get_submesh.
        multiplier_type = get_opaque_type_name(OpaqueMultiplier)

        self.lib.define(
            f"derive_multiplier({multiplier_type} parent, float scale)"
            f" -> {multiplier_type}",
        )

        @torch.library.impl(
            "_TestOpaqueObject::derive_multiplier",
            "CompositeExplicitAutograd",
            lib=self.lib,
        )
        def derive_impl(parent, scale):
            # [FUNCTION DOC] HELPER: Real implementation of the derive op: a new multiplier with
            # the parent multiplier times the scale.
            return OpaqueMultiplier(parent.multiplier * scale)

        @torch.library.register_fake(
            "_TestOpaqueObject::derive_multiplier", lib=self.lib
        )
        def derive_fake(parent, scale):
            # [FUNCTION DOC] HELPER: Fake implementation of the derive op: a multiplier of 0.0.
            return OpaqueMultiplier(0.0)

        parent = OpaqueMultiplier(3.0)
        # The "child" is derived at eager time before tracing.
        # It's NOT passed as an input — it's captured in a closure,
        # simulating how DeviceMesh submeshes are captured in DTensor
        # backward closures.  This forces _try_reconstruct_opaque to
        # be called when make_fx encounters the child.
        child = OpaqueMultiplier(parent.multiplier * 0.5)
        child._parent = parent
        child._scale = 0.5

        from torch._library.opaque_object import _OPAQUE_TYPES

        original_reconstruct_fn = _OPAQUE_TYPES[OpaqueMultiplier].reconstruct_fn

        def multiplier_reconstruct_fn(obj, get_tracked_proxy, tracer):
            # [FUNCTION DOC] HELPER: Reconstruct hook installed on the multiplier type. Returns
            # None if the object has no recorded parent or if the parent has no tracked proxy;
            # otherwise calls the derive op on the parent proxy and the stored scale and returns
            # the resulting proxy.
            if not hasattr(obj, "_parent"):
                return None
            parent_proxy = get_tracked_proxy(obj._parent)
            if parent_proxy is None:
                return None
            return torch.ops._TestOpaqueObject.derive_multiplier(
                parent_proxy, obj._scale
            )

        _OPAQUE_TYPES[OpaqueMultiplier].reconstruct_fn = multiplier_reconstruct_fn
        try:
            # child is captured from the closure, NOT an input
            def fn(parent, x):
                # [FUNCTION DOC] HELPER: Traced function applying the scaling op to the closed-
                # over child multiplier and x; the parent argument is unused.
                return torch.ops._TestOpaqueObject.mul_with_scale(child, x)

            fake_mode = FakeTensorMode(shape_env=ShapeEnv())
            with fake_mode:
                fake_x = torch.randn(4)

            gm = make_fx(fn, tracing_mode="fake")(parent, fake_x)

            # Find the derive_multiplier node (created by reconstruct_fn)
            derive_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "call_function" and "derive_multiplier" in str(n.target)
            ]
            self.assertGreater(len(derive_nodes), 0, "No derive_multiplier node found")

            for node in derive_nodes:
                self.assertIn(
                    "val",
                    node.meta,
                    lambda msg: f"{msg}\nNode {node.name} created via reconstruct_fn is missing "
                    f"meta['val']. This would cause the partitioner to fail "
                    f"with 'Expected {node.name} to be a tensor'.",
                )
                self.assertTrue(
                    is_opaque_node(node),
                    lambda msg: f"{msg}\nNode {node.name} should be classified as opaque",
                )
        finally:
            _OPAQUE_TYPES[OpaqueMultiplier].reconstruct_fn = original_reconstruct_fn

    def test_partitioner_must_save_opaque_node(self):
        # [FUNCTION DOC] Builds a joint graph by hand with an opaque placeholder tagged as must-
        # save by checkpoint policy and runs the default partitioner with one forward output. The
        # partitioner must not raise, and the forward module must contain at least one opaque
        # placeholder node (the opaque object is passed through rather than saved as a tensor).
        """Opaque nodes tagged MUST_SAVE go to saved_opaque_nodes.

        When activation checkpointing tags an opaque node with
        CheckpointPolicy.MUST_SAVE, the default_partition function
        must route it to saved_opaque_nodes (not saved_values).
        Otherwise the runtime assertion in save_from_forward will
        fail because it expects all saved_values to be Tensors.
        """
        from functorch.compile import default_partition
        from torch._functorch._aot_autograd.graph_compile import is_opaque_node
        from torch.utils.checkpoint import CheckpointPolicy

        m = OpaqueMultiplier(2.0)

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

        # Build a joint fwd+bwd graph where the opaque node is tagged
        # MUST_SAVE — simulates activation checkpointing.
        graph = torch.fx.Graph()
        m_node = graph.placeholder("m")
        m_node.meta["val"] = m
        x_node = graph.placeholder("x")
        with fake_mode:
            x_node.meta["val"] = torch.randn(3)

        mul_node = graph.call_function(
            torch.ops._TestOpaqueObject.mul_with_scale.default,
            (m_node, x_node),
        )
        with fake_mode:
            mul_node.meta["val"] = torch.randn(3)

        # Tag the opaque node as MUST_SAVE
        m_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE

        # Create backward portion: identity backward
        bw_node = graph.call_function(torch.ops.aten.mul.Scalar, (mul_node, 1.0))
        with fake_mode:
            bw_node.meta["val"] = torch.randn(3)

        graph.output((mul_node, bw_node))
        joint_gm = torch.fx.GraphModule({}, graph)

        # Run default_partition — this should NOT raise even though
        # the opaque node is tagged MUST_SAVE.
        num_fwd_outputs = 1
        fw_module, bw_module = default_partition(
            joint_gm, [], num_fwd_outputs=num_fwd_outputs
        )

        # Verify the opaque node ended up in the forward graph as
        # a pass-through (not as a saved tensor that would fail the
        # save_from_forward assertion).
        fw_opaque_nodes = [
            n
            for n in fw_module.graph.nodes
            if n.op == "placeholder" and is_opaque_node(n)
        ]
        self.assertGreater(
            len(fw_opaque_nodes),
            0,
            "Forward graph should have opaque placeholder nodes",
        )

    def test_hoisted_opaque_with_graphpickler(self):
        # [FUNCTION DOC] Skipped without dill. Compiles a function using the hoisted-string op
        # with the eager recording backend, finds the placeholder holding a fake script
        # object, pickles the graph with the graph pickler and loads it with a fresh fake
        # mode, then checks that the restored fake script object wraps a HoistedString with
        # the string value double.
        if import_dill() is None:
            self.skipTest("dill not available")

        backend = EagerAndRecordGraphs()

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply the hoisted-string op to x with a hoisted
            # double string.
            return torch.ops.mylib.op_with_string(x, HoistedString("double"))

        compiled = torch.compile(fn, fullgraph=True, backend=backend)
        compiled(torch.randn(3, 4))

        self.assertEqual(len(backend.graphs), 1)
        captured_gm = backend.graphs[0]

        placeholders = [n for n in captured_gm.graph.nodes if n.op == "placeholder"]
        opaque_ph = None
        for ph in placeholders:
            ev = ph.meta.get("example_value")
            if isinstance(ev, FakeScriptObject):
                opaque_ph = ph
                break
        self.assertIsNotNone(opaque_ph, "no placeholder with FakeScriptObject found")

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        options = Options(ops_filter=None, node_metadata_key_filter=None)
        data = GraphPickler.dumps(captured_gm, options)
        restored_gm = GraphPickler.loads(data, fake_mode)

        restored_phs = [n for n in restored_gm.graph.nodes if n.op == "placeholder"]
        restored_opaque = None
        for ph in restored_phs:
            ev = ph.meta.get("example_value")
            if isinstance(ev, FakeScriptObject):
                restored_opaque = ph
                break
        restored_real = restored_opaque.meta["example_value"].real_obj
        self.assertIsInstance(restored_real, HoistedString)
        self.assertEqual(restored_real.val, "double")


instantiate_parametrized_tests(TestOpaqueObject)


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
class TestOpaqueGenerator(TestCase):
    # [CLASS DOC] Test case for generator inputs to tracing, only run when CUDA is available (the
    # whole class is skipped otherwise). Checks make_fx treats generators as opaque values, and
    # checks the metaclass applied to the generator type.
    def test_make_fx_with_generator(self):
        # [FUNCTION DOC] With CUDA: trace with make_fx (real mode) a module calling the graphsafe
        # RNG-state runner around efficient attention with a cloned generator state. Checks the
        # generator placeholder is last and has one user, whose target is the runner. Then traces
        # a deterministic variant (zero dropout), and checks the traced graph output equals eager.
        """make_fx should trace through Generator inputs as opaque values."""
        from torch._prims.rng_prims import graphsafe_run_with_rng_state

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module running efficient attention with dropout via the RNG-
            # state runner using the supplied generator state; returns the first output.
            def forward(self, q, k, v, rng_state):
                # [FUNCTION DOC] HELPER: Run the attention op under the RNG-state runner with the
                # generator state and return only the first output tensor.
                out = graphsafe_run_with_rng_state(
                    torch.ops.aten._scaled_dot_product_efficient_attention.default,
                    q,
                    k,
                    v,
                    None,
                    True,
                    0.1,
                    True,
                    rng_state=rng_state,
                )
                return out[0]

        q = torch.randn(2, 8, 64, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 8, 64, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(2, 8, 64, 32, device="cuda", dtype=torch.float16)
        gen = torch.cuda.default_generators[0].clone_state()

        gm = make_fx(M(), tracing_mode="real")(q, k, v, gen)

        # The last placeholder (generator) should be used by graphsafe_run_with_rng_state
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        gen_placeholder = placeholders[-1]
        self.assertEqual(len(gen_placeholder.users), 1)
        user = next(iter(gen_placeholder.users))
        self.assertIs(user.target, graphsafe_run_with_rng_state)

        # Verify the traced graph produces the same result as eager.
        # Use dropout_p=0.0 so the result is deterministic.
        class M0(torch.nn.Module):
            # [CLASS DOC] HELPER: Module like the previous one but with dropout probability zero,
            # so output is deterministic.
            def forward(self, q, k, v, rng_state):
                # [FUNCTION DOC] HELPER: Run the attention op without dropout under the RNG-state
                # runner and return only the first output tensor.
                out = graphsafe_run_with_rng_state(
                    torch.ops.aten._scaled_dot_product_efficient_attention.default,
                    q,
                    k,
                    v,
                    None,
                    False,
                    0.0,
                    True,
                    rng_state=rng_state,
                )
                return out[0]

        gen1 = torch.cuda.default_generators[0].clone_state()
        gen2 = torch.cuda.default_generators[0].clone_state()
        gm0 = make_fx(M0(), tracing_mode="real")(q, k, v, gen1)
        expected = M0()(q, k, v, gen2)
        gen3 = torch.cuda.default_generators[0].clone_state()
        actual = gm0(q, k, v, gen3)
        self.assertEqual(actual, expected)

    def test_make_fx_randn_with_generator(self):
        # [FUNCTION DOC] With CUDA: trace with make_fx (real mode) a function calling the random-
        # normal op with a CUDA generator. Checks the graph bakes the generator in as an attribute
        # fetch rather than using the placeholder (which has no users).
        """make_fx should trace torch.randn with a Generator input."""

        def fn(a, generator):
            # [FUNCTION DOC] HELPER: Return a twenty-by-twenty random normal tensor drawn from the
            # given generator on the device of a.
            return torch.randn([20, 20], generator=generator, device=a.device)

        gen = torch.Generator("cuda")
        gm = make_fx(fn, tracing_mode="real")(torch.randn(4, device="cuda"), gen)

        # Generator is baked in as a get_attr constant (not a placeholder input)
        # because torch.randn passes it directly to C++ without going through
        # proxy dispatch. The generator placeholder has 0 users.
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(False)),
            """\
class fn(torch.nn.Module):
    def forward(self, a_1: "f32[4]", generator_1):
        _opaque_obj0 = self._opaque_obj0
        randn: "f32[20, 20]" = torch.ops.aten.randn.generator([20, 20], generator = _opaque_obj0, device = device(type='cuda', index=0), pin_memory = False);  _opaque_obj0 = None
        return randn
""",
        )

    def test_set_generator_metaclass_is_idempotent(self):
        # [FUNCTION DOC] With CUDA: checks that calling the C++ generator-metaclass setter a
        # second time with the same metaclass is harmless and the generator type is still an
        # instance of that metaclass.
        """Calling _set_generator_metaclass twice is a no-op, not an error"""
        from torch._custom_class_base import CustomClassBaseMeta

        # Already called during import; second call should be a no-op.
        torch._C._set_generator_metaclass(CustomClassBaseMeta)
        self.assertIsInstance(torch._C.Generator, CustomClassBaseMeta)

    def test_generator_metaclass_is_set(self):
        # [FUNCTION DOC] With CUDA: checks the generator type's metaclass is the custom-class
        # metaclass and its module name is the C extension module.
        """Generator's metaclass should be CustomClassBaseMeta after import"""
        from torch._custom_class_base import CustomClassBaseMeta

        self.assertIsInstance(torch._C.Generator, CustomClassBaseMeta)
        self.assertEqual(torch._C.Generator.__module__, "torch._C")


if __name__ == "__main__":
    run_tests()
