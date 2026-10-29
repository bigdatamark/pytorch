# Owner(s): ["oncall: pt2"]
# [MODULE DOC] Unit tests for the source-emission helpers in
# torch._functorch._aot_autograd.source_emit (emit_value and _emit_importable). Each test checks
# that a value is turned into a Python source expression plus import statements that rebuild an
# equal object, or that unsupported values (live tensors, symbolic sizes, shared mutable objects,
# malformed reduce results) raise NotImplementedError. Defines module-level fixture types, since
# emission rejects locally defined classes. No import-time side effects.
import collections
import dataclasses
import enum
import functools
import math

import torch
from torch._functorch._aot_autograd.source_emit import (
    _emit_importable,
    _REBUILD_HELPER,
    emit_value,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymIntEqByExpr
from torch.testing._internal.common_utils import run_tests, TestCase


def _emit(obj):
    # [FUNCTION DOC] HELPER: Emit source for an object with a fresh import set. Returns a pair of
    # the emitted expression string and the sorted list of import statements collected.
    """Emit source for ``obj`` and return ``(expr, sorted_imports)``."""
    imports: set[str] = set()
    return emit_value(obj, imports), sorted(imports)


def _roundtrip(obj):
    # [FUNCTION DOC] HELPER: Round-trip an object through source: emit it, execute the rebuild
    # helper lines (only when the expression uses the rebuild helper) and the import statements in
    # a fresh namespace, then evaluate the expression there and return the rebuilt object.
    """Emit ``obj`` as source, then exec the imports + ``_rebuild`` helper and eval the
    expression -- the same way the generated standalone module reconstructs it."""
    expr, imports = _emit(obj)
    ns: dict = {}
    if "_rebuild(" in expr:
        exec("\n".join(_REBUILD_HELPER), ns)
    for stmt in imports:
        exec(stmt, ns)
    return eval(expr, ns)


# Module-level fixtures: _emit_importable rejects any object whose __qualname__ carries a
# "<locals>" component, so enums / dataclasses / namedtuples used in the emission tests
# must be defined at module scope to be source-reconstructible.
class _Color(enum.Enum):
    # [CLASS DOC] HELPER: Module-level enum with two members, used by the enum emission test.
    RED = 1
    BLUE = 2


class _IColor(enum.IntEnum):
    # [CLASS DOC] HELPER: Module-level integer enum whose repr is not valid source.
    RED = 1
    BLUE = 2


class _Flag(enum.IntFlag):
    # [CLASS DOC] HELPER: Module-level integer flag enum used to test combined and empty
    # flag values.
    A = 1
    B = 2


class _MyList(list):
    # [CLASS DOC] HELPER: List subclass with no extra behavior; emission must reject it.
    # A builtin-container subclass; emitting it as a plain list would silently drop the
    # subclass, so it must be rejected (via _emit_via_reduce's listitems guard).
    pass


_Point = collections.namedtuple("_Point", ["x", "y"])


@dataclasses.dataclass
class _PlainDC:
    # [CLASS DOC] HELPER: Plain dataclass with two init fields; the simple constructor round
    # trip case.
    a: int
    b: str


@dataclasses.dataclass
class _DerivedDC:
    # [CLASS DOC] HELPER: Dataclass with a non-init field that is recomputed from an init field
    # after init.
    a: int
    derived: int = dataclasses.field(init=False, default=0)

    def __post_init__(self):
        # [FUNCTION DOC] HELPER: Set the derived field to twice the init field, so a constructor
        # call reproduces it.
        # Re-derived from an init field, so a constructor call reproduces it: this still
        # round-trips and must NOT be rejected.
        self.derived = self.a * 2


@dataclasses.dataclass
class _StatefulDC:
    # [CLASS DOC] HELPER: Dataclass with a non-init field whose mutated value cannot be reproduced
    # by the constructor; emission must reject it.
    a: int
    extra: int = dataclasses.field(init=False, default=0)


@dataclasses.dataclass(frozen=True)
class _FrozenDC:
    # [CLASS DOC] HELPER: Frozen dataclass; immutable, so sharing one instance is allowed.
    # Frozen -> immutable, so a shared instance has no aliasing to lose and is exempt from
    # the shared-mutable guard, like a tuple.
    a: int
    b: str


class _MyInt(int):
    # [CLASS DOC] HELPER: Int subclass whose repr looks like a constructor call.
    def __repr__(self):
        # [FUNCTION DOC] HELPER: Return a constructor-style text showing the class name and the
        # integer value.
        return f"_MyInt({int(self)})"


class _ReducesToLoadFromBytes:
    # [CLASS DOC] HELPER: Object whose reduce result delegates to the torch storage byte loader;
    # emission must reject it.
    # A non-Tensor wrapper whose reduce delegates to storage bytes -- emitting it as
    # source would embed raw bytes and require a pickle.loads-equivalent at exec time.
    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC] HELPER: Return a reduce tuple of the storage byte loader and a bytes
        # argument tuple.
        return (torch.storage._load_from_bytes, (b"\x00\x01\x02",))


def _ss_new(cls):
    # [FUNCTION DOC] HELPER: Create a bare instance of the given class without running
    # its initializer.
    return cls.__new__(cls)


class _StateSetterObj:
    # [CLASS DOC] HELPER: Object whose reduce result is the six-element protocol 5 form with a
    # state setter, which the rebuild helper cannot apply; emission must reject it.
    # __reduce_ex__ returns the protocol-5 6-tuple form whose last element is a
    # state_setter; _rebuild cannot apply it, so emission must reject this object.
    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC] HELPER: Return a six-element reduce tuple: factory, class argument, dict
        # state, two Nones and a state setter function that does nothing.
        return (_ss_new, (type(self),), {"x": 1}, None, None, lambda obj, state: None)


def _make_holder(value):
    # [FUNCTION DOC] HELPER: Factory used by the holder class reduce: wraps the value in a
    # new holder.
    return _Holder(value)


class _Holder:
    # [CLASS DOC] HELPER: Value object reducing to a top-level factory call; exercises the generic
    # callable reduce branch. Stores one value and compares equal to holders with an equal value.
    # Reduces to a top-level factory call (the generic-callable branch of
    # _emit_via_reduce: func is neither copyreg.__newobj__ nor __newobj_ex__).
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Store the wrapped value.
        self.value = value

    def __reduce__(self):
        # [FUNCTION DOC] HELPER: Return a reduce tuple of the module-level factory and the stored
        # value as argument.
        return (_make_holder, (self.value,))

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other object is a holder with an equal value.
        return isinstance(other, _Holder) and self.value == other.value


class _NewObjEx:
    # [CLASS DOC] HELPER: Object with a custom new method and extended new-args protocol, so its
    # reduce uses the new-object-ex form with keyword arguments. Holds attributes a and b.
    # __getnewargs_ex__ makes __reduce_ex__(2) use copyreg.__newobj_ex__ with non-empty
    # kwargs, exercising that branch of _emit_via_reduce.
    def __new__(cls, a, b):
        # [FUNCTION DOC] HELPER: Create the instance and set attributes a and b from the arguments.
        obj = object.__new__(cls)
        obj.a = a
        obj.b = b
        return obj

    def __getnewargs_ex__(self):
        # [FUNCTION DOC] HELPER: Return the positional args tuple holding a and the keyword
        # mapping holding b.
        return ((self.a,), {"b": self.b})

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other is the same class and both a and b match.
        return isinstance(other, _NewObjEx) and self.a == other.a and self.b == other.b


class _SlotObj:
    # [CLASS DOC] HELPER: Object using only slots with a single slot x; its reduce state is a pair
    # of dict state and slot state, exercising the slot-state rebuild branch.
    # __slots__ only -> __reduce_ex__(2) yields a (dict_state, slot_state) 2-tuple state,
    # exercising _rebuild's slotstate branch (setattr per slot) -- distinct from the plain
    # __dict__.update branch _NewObjEx hits and the __setstate__ branch a Shard hits.
    __slots__ = ("x",)

    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Store the value in slot x.
        self.x = x

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other is the same class and the slot values match.
        return isinstance(other, _SlotObj) and self.x == other.x


@dataclasses.dataclass(eq=False)
class _EqFalseDC:
    # [CLASS DOC] HELPER: Dataclass with equality comparison disabled, so equality is identity and
    # the constructor round trip check can never pass; emission must fall back to reduce.
    # eq=False -> inherits object.__eq__ (identity), so the constructor round-trip check
    # (rebuilt == obj) can never pass; _emit_value must fall through to _emit_via_reduce
    # and reconstruct it rather than spuriously rejecting a pure value object.
    a: int
    b: str


class _NewObjNoClass:
    # [CLASS DOC] HELPER: Object whose reduce uses the new-object form with an empty argument
    # tuple (no class); emission must reject it.
    # Malformed protocol-2 reduce: copyreg.__newobj__ with an empty args tuple (no class).
    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC] HELPER: Return a reduce tuple of the copyreg new-object function with
        # no arguments.
        import copyreg

        return (copyreg.__newobj__, ())


class _NewObjExBadArity:
    # [CLASS DOC] HELPER: Object whose reduce uses the new-object-ex form with arguments that are
    # not a triple of class, args and kwargs; emission must reject it.
    # Malformed protocol-2 reduce: copyreg.__newobj_ex__ whose args is not a
    # (cls, args, kwargs) triple.
    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC] HELPER: Return a reduce tuple of the new-object-ex function with a one-
        # element argument tuple.
        import copyreg

        return (copyreg.__newobj_ex__, (type(self),))


class _NewObjExBadTriple:
    # [CLASS DOC] HELPER: Object whose new-object-ex reduce has three arguments but a non-mapping
    # kwargs slot; emission must reject the malformed contents.
    # copyreg.__newobj_ex__ with a structurally-3 tuple whose kwargs slot is not a dict;
    # the guard must reject the malformed CONTENTS, not just arity.
    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC] HELPER: Return a reduce tuple of the new-object-ex function with class,
        # empty tuple and None.
        import copyreg

        return (copyreg.__newobj_ex__, (type(self), (), None))


class _ReduceRaises:
    # [CLASS DOC] HELPER: Object whose reduce method itself raises, so it has no usable reduce.
    # __reduce_ex__ itself raises -> the "no usable reduce" reject path.
    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC] HELPER: Always raise RuntimeError.
        raise RuntimeError("nope")


class _ShortReduce:
    # [CLASS DOC] HELPER: Object whose reduce result is a one-element tuple, an unsupported form.
    # __reduce_ex__ returns a 1-tuple (len < 2) -> the "unsupported __reduce__ form" reject.
    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC] HELPER: Return a one-element tuple containing the object type.
        return (object,)


class _ReduceNonTupleArgs:
    # [CLASS DOC] HELPER: Object whose reduce result has a non-tuple args field; emission must
    # reject it.
    # __reduce_ex__ returns a non-tuple args field (the generic-callable branch); must
    # reject cleanly rather than leak a bare TypeError on iteration.
    def __reduce_ex__(self, protocol):
        # [FUNCTION DOC] HELPER: Return a pair of the list type and the integer 5 as the
        # args field.
        return (list, 5)


class TestSourceEmit(TestCase):
    # [CLASS DOC] Tests for source emission: every branch of the value emitter either round-trips
    # to an equal object or raises NotImplementedError with a specific message, so the generated
    # standalone artifact stays auditable and free of pickle loading.
    # Unit coverage of the source-emission helpers: every _emit_value branch round-trips
    # (or raises on a non-source-expressible leaf), so the standalone artifact stays
    # auditable and pickle.loads-free.
    def test_none_and_exact_builtin_scalars(self):
        # [FUNCTION DOC] Checks exact-type scalars (None, bool, int, float, str, bytes, bytearray,
        # complex) emit their literal text with no imports.
        self.assertEqual(_emit(None), ("None", []))
        self.assertEqual(_emit(True), ("True", []))
        self.assertEqual(_emit(42), ("42", []))
        self.assertEqual(_emit(3.5), ("3.5", []))
        self.assertEqual(_emit("ab"), ("'ab'", []))
        self.assertEqual(_emit(b"xy"), ("b'xy'", []))
        self.assertEqual(_emit(bytearray(b"xy")), ("bytearray(b'xy')", []))
        self.assertEqual(_emit(complex(1, 2)), ("(1+2j)", []))

    def test_non_finite_float_and_complex(self):
        # [FUNCTION DOC] Checks infinite and NaN floats and complex numbers emit via the float
        # constructor text and still round-trip (NaN stays NaN, infinite real part preserved).
        self.assertEqual(_emit(float("inf")), ("float('inf')", []))
        self.assertEqual(_emit(float("-inf")), ("float('-inf')", []))
        self.assertEqual(_emit(float("nan")), ("float('nan')", []))
        self.assertTrue(math.isnan(_roundtrip(float("nan"))))
        self.assertEqual(
            _emit(complex(float("inf"), 0.0))[0], "complex(float('inf'), 0.0)"
        )
        rt = _roundtrip(complex(float("inf"), 0.0))
        self.assertTrue(math.isinf(rt.real) and rt.imag == 0.0)

    def test_int_subclass_falls_through_repr(self):
        # [FUNCTION DOC] Checks an int subclass with a constructor-style repr does not use the
        # plain repr branch and round-trips through the reduce path, keeping its subclass type.
        # An int subclass with a constructor-style __repr__ must NOT take the exact-type
        # repr branch (which would emit a NameError / lose its type); it round-trips via
        # the reduce path and keeps its type.
        rt = _roundtrip(_MyInt(5))
        self.assertEqual(rt, 5)
        self.assertIs(type(rt), _MyInt)

    def test_torch_scalar_singletons(self):
        # [FUNCTION DOC] Checks torch dtype, layout and memory format singletons emit as attribute
        # references with an import of torch, and round-trip to the identical object.
        self.assertEqual(_emit(torch.float32), ("torch.float32", ["import torch"]))
        self.assertEqual(_emit(torch.strided), ("torch.strided", ["import torch"]))
        self.assertEqual(
            _emit(torch.contiguous_format),
            ("torch.contiguous_format", ["import torch"]),
        )
        self.assertIs(_roundtrip(torch.float32), torch.float32)

    def test_torch_device_size(self):
        # [FUNCTION DOC] Checks torch device and Size values emit as constructor text with a torch
        # import and round-trip to equal values.
        self.assertEqual(
            _emit(torch.device("cpu")), ("torch.device('cpu')", ["import torch"])
        )
        self.assertEqual(_roundtrip(torch.device("cpu")), torch.device("cpu"))
        self.assertEqual(
            _emit(torch.Size([2, 3])), ("torch.Size([2, 3])", ["import torch"])
        )
        self.assertEqual(_roundtrip(torch.Size([2, 3])), torch.Size([2, 3]))

    def test_importable_class_function_module(self):
        # [FUNCTION DOC] Checks importable classes and modules round-trip to the identical object
        # and that a module emits as its name with an import statement.
        self.assertIs(_roundtrip(torch.nn.Linear), torch.nn.Linear)
        self.assertIs(_roundtrip(math), math)
        self.assertEqual(_emit(math), ("math", ["import math"]))

    def test_importable_rejects_lambda_and_local(self):
        # [FUNCTION DOC] Checks that a lambda and a locally defined function are rejected with
        # NotImplementedError mentioning a local definition.
        with self.assertRaisesRegex(NotImplementedError, "local definition"):
            _emit(lambda x: x)

        def _local():
            # [FUNCTION DOC] HELPER: Local function with no behavior, used as an unemittable
            # local definition.
            pass

        with self.assertRaisesRegex(NotImplementedError, "local definition"):
            _emit(_local)

    def test_enums(self):
        # [FUNCTION DOC] Checks enum, IntEnum members round-trip to the identical member (the
        # integer enum must use the enum branch, since its repr is not valid source).
        self.assertIs(_roundtrip(_Color.RED), _Color.RED)
        self.assertIs(_roundtrip(_Color.BLUE), _Color.BLUE)
        # IntEnum: repr is "<_IColor.RED: 1>" (invalid source), so it must take the enum
        # branch, not the repr branch.
        self.assertIs(_roundtrip(_IColor.RED), _IColor.RED)

    def test_combined_flag_emits_by_value(self):
        # [FUNCTION DOC] Checks a combined flag value, which has no single member name, is rebuilt
        # by value, while single members stay singletons and the empty flag round-trips.
        # A combined Flag member (A | B) has no single member name; emitting by name would
        # produce "Type.None" (a SyntaxError). It must reconstruct by value instead.
        combined = _Flag.A | _Flag.B
        self.assertEqual(_roundtrip(combined), combined)
        self.assertIs(_roundtrip(_Flag.A), _Flag.A)  # singleton still by-name
        self.assertEqual(_roundtrip(_Flag(0)), _Flag(0))  # empty flag

    def test_functools_partial(self):
        # [FUNCTION DOC] Checks functools.partial objects round-trip: baked function, positional
        # and keyword arguments are applied on call, and the emitted imports include functools.
        # _roundtrip rebuilds the partial object itself; invoking it then applies the
        # baked func/args/keywords.
        p = _roundtrip(functools.partial(_PlainDC, b="z"))
        self.assertEqual(p(a=1), _PlainDC(1, "z"))
        _expr, imports = _emit(functools.partial(int, "10", base=2))
        self.assertIn("import functools", imports)
        self.assertEqual(_roundtrip(functools.partial(int, "10", base=2))(), 2)

    def test_containers(self):
        # [FUNCTION DOC] Checks tuple, list and dict literal formatting (including one-element and
        # empty tuples) and a nested container round trip.
        self.assertEqual(_emit((1, 2)), ("(1, 2)", []))
        self.assertEqual(_emit((1,)), ("(1,)", []))
        self.assertEqual(_emit(()), ("()", []))
        self.assertEqual(_emit([1, 2]), ("[1, 2]", []))
        self.assertEqual(_emit([]), ("[]", []))
        self.assertEqual(_emit({"a": 1}), ("{'a': 1}", []))
        self.assertEqual(_roundtrip({"k": [1, 2, (3,)]}), {"k": [1, 2, (3,)]})

    def test_namedtuple(self):
        # [FUNCTION DOC] Checks a namedtuple round-trips to an equal value of the same
        # namedtuple type.
        rt = _roundtrip(_Point(1, 2))
        self.assertEqual(rt, _Point(1, 2))
        self.assertIs(type(rt), _Point)

    def test_set_canonical_ordering_is_deterministic(self):
        # [FUNCTION DOC] Checks sets and frozensets emit with elements in a canonical order,
        # sorted by emitted source text (lexicographic, so ten sorts before two), independent of
        # insertion order, and mixed-type elements sort without error.
        # Set iteration order is not byte-stable across processes; emission sorts to a
        # canonical order so the artifact is reproducible.
        self.assertEqual(_emit({3, 1, 2})[0], "set([1, 2, 3])")
        self.assertEqual(_emit(frozenset({3, 1, 2}))[0], "frozenset([1, 2, 3])")
        self.assertEqual(_emit({3, 1, 2})[0], _emit({2, 3, 1})[0])
        # Sorting is by EMITTED SOURCE (a lexicographic string sort), not numeric, so a
        # multi-digit int set orders "1" < "10" < "2" -- still deterministic. Lock that
        # order so the canonicalization key can't silently regress to numeric/iteration.
        self.assertEqual(_emit({1, 2, 10})[0], "set([1, 10, 2])")
        # Unorderable elements (int vs str) also sort by emitted source; a numeric sort
        # would raise a TypeError here.
        self.assertEqual(_emit({1, "a"})[0], "set(['a', 1])")
        self.assertEqual(_emit({1, "a", "b"})[0], _emit({"b", 1, "a"})[0])

    def test_plain_dataclass_round_trips(self):
        # [FUNCTION DOC] Checks a plain dataclass round-trips to an equal instance.
        rt = _roundtrip(_PlainDC(1, "z"))
        self.assertEqual(rt, _PlainDC(1, "z"))

    def test_dataclass_with_derived_post_init_round_trips(self):
        # [FUNCTION DOC] Checks a dataclass with a field derived in post-init round-trips, with
        # the derived value recomputed as ten for an init value of five.
        rt = _roundtrip(_DerivedDC(5))
        self.assertEqual(rt, _DerivedDC(5))
        self.assertEqual(rt.derived, 10)

    def test_stateful_dataclass_is_rejected(self):
        # [FUNCTION DOC] Checks a dataclass whose non-init field was mutated to an unreproducible
        # value is rejected with NotImplementedError saying it does not round-trip.
        # A non-init field mutated to a value the constructor cannot reproduce makes the
        # rebuilt instance compare unequal: emitting only the init fields would silently
        # drop that state, so it must raise.
        obj = _StatefulDC(5)
        obj.extra = 99
        with self.assertRaisesRegex(NotImplementedError, "does not round-trip"):
            _emit(obj)

    def test_live_tensor_and_storage_rejected(self):
        # [FUNCTION DOC] Checks live tensors, untyped storage and typed storage are rejected with
        # NotImplementedError naming the live object kind.
        with self.assertRaisesRegex(NotImplementedError, "live Tensor"):
            _emit(torch.zeros(2))
        with self.assertRaisesRegex(NotImplementedError, "live UntypedStorage"):
            _emit(torch.zeros(3).untyped_storage())
        # A TypedStorage (the legacy .storage() type) is the third arm of the same
        # live-storage guard: baking it would embed raw bytes and need pickle.loads.
        with self.assertRaisesRegex(NotImplementedError, "live TypedStorage"):
            _emit(torch.zeros(2).storage())

    def test_symbolic_symint_rejected(self):
        # [FUNCTION DOC] Checks an unbacked symbolic integer is rejected with NotImplementedError
        # saying that precompile specializes to static shapes.
        # A SymInt with no concrete value cannot be baked: precompile specializes to
        # static shapes, so a symbolic size has no source literal. Build an unbacked
        # symint (maybe_as_int() is None) and confirm the static-shapes reject fires.
        symint = ShapeEnv().create_unbacked_symint()
        self.assertIsNone(symint.node.maybe_as_int())
        static = "specializes to static shapes"
        with self.assertRaisesRegex(NotImplementedError, static):
            emit_value(symint, set())

    def test_symbolic_view_metadata_rejected(self):
        # [FUNCTION DOC] Checks the equality wrapper around a still-symbolic integer expression is
        # rejected with the same static shapes NotImplementedError.
        # SymIntEqByExpr wraps a sympy expr that must be a concrete integer to bake; a
        # symbolic one (its .val is non-Integer) hits the same static-shapes reject. The
        # constructor coerces ints to sympy.Integer, so feed it a symbolic SymInt to get
        # a non-Integer .val (sympy.Symbol cannot be passed directly -- it is coerced).
        obj = SymIntEqByExpr(ShapeEnv().create_unbacked_symint())
        self.assertFalse(getattr(obj.val, "is_Integer", False))
        static = "specializes to static shapes"
        with self.assertRaisesRegex(NotImplementedError, static):
            emit_value(obj, set())

    def test_view_meta_sequence_round_trips(self):
        # [FUNCTION DOC] Builds a view replay sequence from parts (one resize view metadata and a
        # metadata key with concrete wrapped sizes) and checks the emitted text uses the parts
        # factory, and that it round-trips to a sequence with equal metadata and equal view
        # metadata tuples.
        # The view-replay-recipe branches (a ViewMeta via as_tuple, the ViewMetaSequence
        # via its _from_parts factory, and the concrete SymIntEqByExpr bake inside its
        # MetadataKey) are the headline metadata this module reconstructs. Build a
        # ViewMetaSequence straight from parts (no live FunctionalTensor needed) and
        # confirm it emits a _from_parts expression that round-trips to an equal recipe.
        from torch._C import _functionalization as _F
        from torch._functorch._aot_autograd.functional_utils import (
            MetadataKey,
            ViewMetaSequence,
        )

        view_meta = _F.resize__ViewMeta((True, [3, 4, 5]))
        meta = MetadataKey(
            size=(SymIntEqByExpr(2),),
            layout=torch.strided,
            is_sparse=False,
            stride=(SymIntEqByExpr(1),),
            storage_offset=SymIntEqByExpr(0),
            is_conj=False,
            is_neg=False,
        )
        vms = ViewMetaSequence._from_parts([view_meta], meta)
        expr, _imports = _emit(vms)
        self.assertIn("ViewMetaSequence._from_parts(", expr)
        self.assertIn("ViewMeta(", expr)
        rt = _roundtrip(vms)
        self.assertIsInstance(rt, ViewMetaSequence)
        self.assertEqual(rt.metadata, meta)
        self.assertEqual([v.as_tuple() for v in rt.sequence], [view_meta.as_tuple()])

    def test_symbolic_symbool_symfloat_rejected(self):
        # [FUNCTION DOC] Checks unbacked symbolic bool and float values are rejected with the
        # static shapes NotImplementedError.
        # SymBool / SymFloat get the same static-shapes treatment as SymInt: a still-symbolic
        # value cannot be baked and raises the clear static-shapes message (rather than
        # falling into the reduce path with an opaque ValueRanges error).
        shape_env = ShapeEnv()
        static = "specializes to static shapes"
        sym_bool = shape_env.create_unbacked_symbool()
        self.assertIsNone(sym_bool.node.maybe_as_bool())
        with self.assertRaisesRegex(NotImplementedError, static):
            emit_value(sym_bool, set())
        sym_float = shape_env.create_unbacked_symfloat()
        self.assertIsNone(sym_float.node.maybe_as_float())
        with self.assertRaisesRegex(NotImplementedError, static):
            emit_value(sym_float, set())

    def test_concrete_symbool_symfloat_baked(self):
        # [FUNCTION DOC] Checks a symbolic bool and float that fold to concrete values are emitted
        # as literals and round-trip to True and 2.5.
        # A SymBool / SymFloat that folds to a concrete value IS bakeable (mirroring SymInt),
        # emitted as the literal -- previously these fell through and were wrongly rejected.
        s = ShapeEnv().create_unbacked_symint()
        sym_bool = (s - s) == 0
        self.assertIs(sym_bool.node.maybe_as_bool(), True)
        self.assertEqual(_roundtrip(sym_bool), True)
        sym_float = s * 0.0 + 2.5
        self.assertEqual(sym_float.node.maybe_as_float(), 2.5)
        self.assertEqual(_roundtrip(sym_float), 2.5)

    def test_reduce_to_load_from_bytes_rejected(self):
        # [FUNCTION DOC] Regression: an object whose reduce function is the storage byte loader is
        # rejected with NotImplementedError rather than baking raw bytes.
        # Regression: the previous guard compared the __reduce_ex__ METHOD against
        # _load_from_bytes and never fired; the callable is the reduce RESULT's func.
        # A wrapper whose reduce is _load_from_bytes must be rejected, not silently emit
        # raw bytes + a pickle.loads-equivalent.
        with self.assertRaisesRegex(NotImplementedError, "_load_from_bytes"):
            _emit(_ReducesToLoadFromBytes())

    def test_state_setter_reduce_rejected(self):
        # [FUNCTION DOC] Regression: an object whose reduce result carries a state setter is
        # rejected with NotImplementedError mentioning state_setter.
        # Regression: a protocol-5 6-tuple reduce carries a state_setter _rebuild cannot
        # apply; emitting _rebuild(base, state) would install state via the wrong
        # mechanism, so reject it.
        with self.assertRaisesRegex(NotImplementedError, "state_setter"):
            _emit(_StateSetterObj())

    def test_self_referential_container_rejected(self):
        # [FUNCTION DOC] Checks self-referential lists and dicts are rejected
        # with NotImplementedError.
        a = [1, 2]
        a.append(a)
        with self.assertRaisesRegex(NotImplementedError, "self-referential"):
            _emit(a)
        d: dict = {}
        d["self"] = d
        with self.assertRaisesRegex(NotImplementedError, "self-referential"):
            _emit(d)

    def test_shared_mutable_rejected(self):
        # [FUNCTION DOC] Checks a mutable object reached twice as siblings (list, dict, set,
        # bytearray, reduce based object, non-frozen dataclass) is rejected with
        # NotImplementedError mentioning shared mutable, to avoid silently losing aliasing.
        # A mutable object reached twice (not on the recursion path -- as siblings) would
        # be emitted as two independent literals, silently dropping the shared-identity
        # aliasing (mutating one reconstructed copy would no longer affect the other). Fail
        # loud instead, consistent with the module's other reject-don't-mis-bake guards.
        a = [1, 2]
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([a, a])
        d = {"k": 1}
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit((d, d))
        s = {1, 2}
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([s, s])
        # A bytearray is emitted as a repr literal but is mutable, so a shared one is
        # rejected too rather than emitted as two independent literals.
        ba = bytearray(b"xy")
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([ba, ba])
        # An opaque reduce value object shared across positions is caught the same way,
        # via _emit_via_reduce's guard rather than the container guard.
        h = _Holder(3)
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([h, h])
        # A shared non-frozen dataclass is mutable too: emitting it as two constructor
        # calls would drop the aliasing, so the dataclass branch also routes it here.
        dc = _PlainDC(1, "z")
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([dc, dc])

    def test_shared_frozen_dataclass_is_not_a_cycle(self):
        # [FUNCTION DOC] Checks a frozen dataclass shared at two positions is emitted fine and
        # round- trips, because immutable objects have no aliasing to lose.
        # A frozen dataclass is immutable, so a shared instance has no aliasing to lose and
        # emits fine as two independent literals -- exempt from the shared-mutable guard.
        fdc = _FrozenDC(1, "z")
        expr, _ = _emit([fdc, fdc])
        self.assertIn("_FrozenDC", expr)
        rt = _roundtrip([fdc, fdc])
        self.assertEqual(rt, [_FrozenDC(1, "z"), _FrozenDC(1, "z")])

    def test_repeated_leaf_is_not_a_cycle(self):
        # [FUNCTION DOC] Checks repeated sibling scalars and a repeated immutable tuple are not
        # mistaken for cycles or shared mutable objects.
        # The cycle guard tracks only ancestors on the recursion path, not siblings, so a
        # leaf repeated across sibling positions is not a false cycle -- for a scalar...
        self.assertEqual(_emit([0, 0, 0]), ("[0, 0, 0]", []))
        # ...and for a shared IMMUTABLE object repeated as siblings (only shared MUTABLE
        # objects are rejected; an immutable tuple has no aliasing to lose).
        inner = (1, 2)
        self.assertEqual(_emit([inner, inner]), ("[(1, 2), (1, 2)]", []))

    def test_container_subclass_rejected(self):
        # [FUNCTION DOC] Checks a list subclass is rejected with NotImplementedError mentioning
        # container subclass instead of being downcast.
        # A builtin-container subclass must not be silently downcast to its base type; it
        # falls through to _emit_via_reduce, whose listitems guard rejects it.
        with self.assertRaisesRegex(NotImplementedError, "container subclass"):
            _emit(_MyList([1, 2]))

    def test_emit_via_reduce_round_trips_opaque_object(self):
        # [FUNCTION DOC] Checks an opaque object with a new-object reduce and dict state (a
        # DTensor Shard placement) round-trips; skipped if the placement types cannot be imported.
        # An opaque value object with a copyreg.__newobj__ reduce + dict state rebuilds via
        # cls.__new__ + _rebuild, emitted as source (no pickle.loads).
        try:
            from torch.distributed.tensor.placement_types import Shard
        except Exception:
            self.skipTest("DTensor placement_types unavailable")
        rt = _roundtrip(Shard(2))
        self.assertEqual(rt, Shard(2))

    def test_emit_via_reduce_generic_callable(self):
        # [FUNCTION DOC] Checks an object reducing to a top-level factory round-trips to an
        # equal object.
        # An object whose reduce is a plain top-level factory (not copyreg.__newobj__)
        # rebuilds via that factory call, emitted as source.
        rt = _roundtrip(_Holder(7))
        self.assertEqual(rt, _Holder(7))

    def test_emit_via_reduce_newobj_ex(self):
        # [FUNCTION DOC] Checks an object using the new-object-ex reduce with keyword arguments
        # round- trips.
        # __getnewargs_ex__ drives the copyreg.__newobj_ex__ branch (cls.__new__ with
        # positional + keyword args) plus _rebuild for the dict state.
        rt = _roundtrip(_NewObjEx(1, b=2))
        self.assertEqual(rt, _NewObjEx(1, b=2))

    def test_emit_via_reduce_slots_state(self):
        # [FUNCTION DOC] Checks a slots-only object round-trips, which only works if slot state is
        # applied because the new method does not set the slot.
        # A __slots__ object reduces to a (dict, slots) 2-tuple state, exercising _rebuild's
        # slotstate branch. _SlotObj.__new__ does NOT set x, so the round-trip is correct
        # only if _rebuild applies the slotstate -- making that branch load-bearing here.
        rt = _roundtrip(_SlotObj(5))
        self.assertEqual(rt, _SlotObj(5))
        self.assertEqual(rt.x, 5)

    def test_eq_false_dataclass_round_trips_via_reduce(self):
        # [FUNCTION DOC] Checks a dataclass with identity equality falls back to reduce and
        # rebuilds an instance of the same type, compared field by field.
        # An eq=False dataclass has identity __eq__, so the constructor round-trip check can
        # never pass; _emit_value must fall through to _emit_via_reduce and reconstruct it
        # (as source) instead of rejecting a pure value object. Assert by fields, since ==
        # is identity for this type.
        obj = _EqFalseDC(7, "z")
        rt = _roundtrip(obj)
        self.assertIs(type(rt), _EqFalseDC)
        self.assertEqual((rt.a, rt.b), (7, "z"))

    def test_newobj_reduce_without_class_rejected(self):
        # [FUNCTION DOC] Checks a new-object reduce with no class argument is rejected with
        # NotImplementedError mentioning no class argument.
        # copyreg.__newobj__ with an empty args tuple has no class to construct; reject
        # cleanly (NotImplementedError) rather than raising a bare IndexError.
        with self.assertRaisesRegex(NotImplementedError, "no class argument"):
            _emit(_NewObjNoClass())

    def test_newobj_ex_reduce_bad_arity_rejected(self):
        # [FUNCTION DOC] Checks a new-object-ex reduce whose args are not a class, args, kwargs
        # triple is rejected with NotImplementedError.
        # copyreg.__newobj_ex__ whose args is not a (cls, args, kwargs) triple must reject
        # cleanly rather than raising a bare ValueError on unpack.
        with self.assertRaisesRegex(NotImplementedError, "cls, args, kwargs"):
            _emit(_NewObjExBadArity())

    def test_newobj_ex_reduce_bad_triple_contents_rejected(self):
        # [FUNCTION DOC] Checks a three-element new-object-ex argument with a non-mapping kwargs
        # slot is rejected with the same NotImplementedError.
        # A structurally-3 tuple whose args/kwargs slots are the wrong types must also
        # reject cleanly, not leak a bare AttributeError/TypeError deeper in reconstruction.
        with self.assertRaisesRegex(NotImplementedError, "cls, args, kwargs"):
            _emit(_NewObjExBadTriple())

    def test_reduce_raises_rejected(self):
        # [FUNCTION DOC] Checks an object whose reduce method raises is rejected with
        # NotImplementedError saying there is no usable reduce.
        # When __reduce_ex__ itself raises, the object has no usable reduce; surface a clean
        # NotImplementedError naming the cause.
        with self.assertRaisesRegex(NotImplementedError, "no usable reduce"):
            _emit(_ReduceRaises())

    def test_short_reduce_form_rejected(self):
        # [FUNCTION DOC] Checks a reduce result shorter than two elements is rejected with
        # NotImplementedError about an unsupported reduce form.
        # A reduce tuple of len < 2 is an unsupported form; reject cleanly.
        with self.assertRaisesRegex(NotImplementedError, "unsupported __reduce__ form"):
            _emit(_ShortReduce())

    def test_reduce_non_tuple_args_rejected(self):
        # [FUNCTION DOC] Checks a reduce result whose args field is not a tuple is
        # rejected with NotImplementedError.
        # A reduce whose args field is not a tuple must reject cleanly (uniform with the
        # __newobj__ / __newobj_ex__ guards) rather than leak a bare TypeError.
        with self.assertRaisesRegex(NotImplementedError, "args field is not a tuple"):
            _emit(_ReduceNonTupleArgs())

    def test_emit_importable_rejects_non_round_tripping(self):
        # [FUNCTION DOC] Checks the importable-object emitter rejects a torch builtin whose
        # qualified name does not round-trip through import, raising NotImplementedError.
        # torch.add is a builtin whose __qualname__ does not round-trip via importlib.
        with self.assertRaisesRegex(NotImplementedError, "does not.*round-trip"):
            _emit_importable(torch.add, set())


if __name__ == "__main__":
    run_tests()
