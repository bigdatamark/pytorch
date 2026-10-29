# [MODULE DOC] Custom pickling of FX graphs and GraphModules (GraphPickler) for caching and
# precompilation. A Pickler subclass intercepts fake tensors, graph modules, ops, shape
# environments, symbolic ints, tracing contexts, weakrefs, opaque script objects and torch-numpy
# functions and replaces them with small "pickle data" helper classes; unpickling needs a
# FakeTensorMode, which is passed to those helpers through a persistent-id token. Also holds
# the Options dataclass (op and metadata filters). Import-time side effect: uses dill in place
# of pickle when dill is importable.
import contextlib
import dataclasses
import importlib
import io
import itertools
import pickle
import weakref
from abc import abstractmethod
from collections.abc import Callable, Generator
from typing import Any, NewType, TypeVar
from typing_extensions import override, Self

from torch.utils._import_utils import import_dill


dill = import_dill()
if dill is not None:
    pickle = dill

import torch
import torch.utils._pytree as pytree
from torch._guards import TracingContext
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    is_fake_tensor,
    Tensor,
)
from torch._subclasses.meta_utils import (
    MetaConverter,
    MetaTensorDesc,
    MetaTensorDescriber,
)
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._mode_utils import no_dispatch


_SymNodeT = TypeVar("_SymNodeT", torch.SymInt, torch.SymFloat)


def _ops_filter_safe(name: str) -> bool:
    # [FUNCTION DOC] HELPER: Default op filter. True when the op name starts with the aten or
    # fbgemm op namespace prefixes, i.e. ops that any machine with PyTorch can resolve.
    """
    An ops filter which allows pickle-safe ops. Pickle-safe ops are built-in
    ones where it will be possible to unpickle on any machine which has PyTorch.
    """
    # TODO: This list is pretty pessimistic right now. What's the full list?
    return name.startswith(
        (
            "torch.ops.aten",
            "torch.ops.fbgemm",
        )
    )


def _node_metadata_key_filter_safe(key: str) -> bool:
    # [FUNCTION DOC] HELPER: Default node-metadata filter. False for the keys holding source
    # function stacks and module stacks (which reference unpicklable objects), true otherwise.
    """
    A metadata filter which allows pickle-safe node metadata. These often times contain
    stacks with pointers to unserializable objects, so we clear them out.
    """
    return key not in ["source_fn_stack", "nn_module_stack", "fwd_source_fn_stack"]


@dataclasses.dataclass
class Options:
    # [CLASS DOC] Settings for GraphPickler: ops_filter (predicate on op name; ops failing it
    # make the cache-bypass validator signal bypass; None allows all), node_metadata_key_filter
    # (predicate on metadata key; keys failing it are dropped; None keeps all), and
    # ignore_raw_node (replace stray Node objects with None instead of raising). Defaults are
    # the pickle-safe filters.
    # A filter for which ops will cause the pickler to raise a
    # BypassFxGraphCache exception. If None then all ops are allowed.
    ops_filter: Callable[[str], bool] | None = _ops_filter_safe
    node_metadata_key_filter: Callable[[str], bool] | None = (
        _node_metadata_key_filter_safe
    )
    # If True, raw torch.fx.Node objects encountered during pickling will be
    # silently replaced with None instead of raising an AssertionError.
    ignore_raw_node: bool = False


def _unpickle_as_none() -> None:
    # [FUNCTION DOC] HELPER: Unpickle callable that always yields None.
    return None


def _unpickle_as_weakref(referent: object) -> weakref.ref[object]:
    # [FUNCTION DOC] HELPER: Unpickle callable that builds a new weak reference to the already
    # unpickled referent.
    return weakref.ref(referent)


def _unpickle_as_dead_weakref() -> Callable[[], None]:
    # [FUNCTION DOC] HELPER: Unpickle callable returning a function that takes no arguments and
    # returns None, imitating a weak reference whose referent is gone.
    return lambda: None


@contextlib.contextmanager
def patch_pytree_map_over_slice() -> Generator[None]:
    # [FUNCTION DOC] Context manager that temporarily registers slice objects as pytree nodes
    # (children are start, stop, step) so tree maps can traverse them while pickling. If
    # slice is already registered it does nothing; otherwise the registration is removed on
    # exit. Mutates the global pytree registry; not thread safe.
    if slice in pytree.SUPPORTED_NODES:
        yield
        return

    pytree._private_register_pytree_node(
        slice, lambda x: ([x.start, x.stop, x.step], None), lambda x, c: slice(*x)
    )

    try:
        yield
    finally:
        pytree._deregister_pytree_node(slice)


# pyrefly: ignore [invalid-inheritance]
class GraphPickler(pickle.Pickler):
    # [CLASS DOC] Pickler specialised for FX graphs. Holds the Options, an opaque token object
    # that stands in for the unpickling state (written as a persistent id), and a shared
    # tensor describer so views and duplicates stay consistent within one pickle. Entry points:
    # dumps, loads (static, needs a FakeTensorMode), and debug_dumps for locating the first
    # unpicklable part. Special types are routed through reducer_override.
    """
    GraphPickler is a Pickler which helps pickling fx graph - in particular
    GraphModule.
    """

    def __init__(self, file: io.BytesIO, options: Options | None = None) -> None:
        # [FUNCTION DOC] Create a pickler writing to the given byte stream (by-reference mode
        # when dill is used). Stores the options (defaults if None), creates the unpickle-state
        # token and a tensor describer that does not copy data.
        if dill is not None:
            super().__init__(file, byref=True)
        else:
            super().__init__(file)
        self.options = options or Options()

        # This abomination is so we can pass external decoding state to the
        # unpickler functions. We serialize _unpickle_state as a persistent
        # external item and when we deserialize it we return the common state
        # object.
        self._unpickle_state = _UnpickleStateToken(object())

        # This is used to describe tensors. It needs to be common across the
        # pickle so that duplicates and views are properly handled.
        self._meta_tensor_describer = MetaTensorDescriber(copy_data=False)

    _PASSTHROUGH_TYPES = frozenset({int, float, str, bytes, bool, type(None)})

    @override
    # pyrefly: ignore [bad-override]
    def reducer_override(
        self, obj: object
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        # [FUNCTION DOC] Pickle hook choosing a custom reduction per object type. Primitive
        # types use default pickling. Fake tensors, graph modules, operators, shape envs,
        # symbolic ints and tracing contexts delegate to their pickle-data helper. Opaque script
        # objects holding value-type constants use default pickling, others become None. Weak
        # references are rebuilt (live) or become dead stand-ins. A raw Node raises an assertion
        # error unless ignore_raw_node is set (then None). Torch-numpy functions are encoded
        # by name; anything else returns NotImplemented for default behavior.
        # KNOWN ISSUE: the return annotation omits the NotImplemented result.
        # This function is supposed to return either NotImplemented (meaning to
        # do the default pickle behavior) or a pair of (unpickle callable, data
        # to pass to unpickle).

        # We could instead teach individual classes how to pickle themselves but
        # that has a few problems:
        #
        #   1. If we have some special needs (maybe for this use-case we don't
        #      want to fully serialize every field) then we're adding private
        #      details to a public interface.
        #
        #   2. If we need to have some common shared data (such as a
        #      FakeTensorMode) which is passed to each value it's harder to
        #      support.

        # These are the types that need special handling. See the individual
        # *PickleData classes for details on pickling that particular type.
        if type(obj) in self._PASSTHROUGH_TYPES:
            return NotImplemented

        if is_fake_tensor(obj):
            return _TensorPickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch.fx.GraphModule):
            return _GraphModulePickleData.reduce_helper(self, obj)
        elif isinstance(obj, (torch._ops.OperatorBase, torch._ops.OpOverloadPacket)):
            return _OpPickleData.reduce_helper(self, obj)
        elif isinstance(obj, ShapeEnv):
            return _ShapeEnvPickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch.SymInt):
            return _SymNodePickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch._guards.TracingContext):
            return _TracingContextPickleData.reduce_helper(self, obj)
        elif isinstance(obj, FakeScriptObject):
            from torch._library.opaque_object import is_opaque_constant_type

            real_obj = object.__getattribute__(obj, "real_obj")
            if real_obj is not None and is_opaque_constant_type(type(real_obj)):
                # Use default pickling; value-type opaques are picklable.
                return NotImplemented
            # Reference-type FakeScriptObjects can't be default-pickled.
            return (_unpickle_as_none, ())
        elif isinstance(obj, weakref.ref):
            # Serialize weakrefs properly: if the referent is alive,
            # serialize it and reconstruct the weakref on unpickle.
            # If the referent is dead, unpickle as a dead-weakref-like callable.
            referent = obj()
            if referent is not None:
                return (_unpickle_as_weakref, (referent,))
            else:
                return (_unpickle_as_dead_weakref, ())
        else:
            # We should never get a raw Node!
            if isinstance(obj, torch.fx.Node):
                if self.options.ignore_raw_node:
                    return (_unpickle_as_none, ())
                raise AssertionError("Unexpected raw Node during pickling")
            if reduce := _TorchNumpyPickleData.reduce_helper(self, obj):
                return reduce

            # returning `NotImplemented` causes pickle to revert to the default
            # behavior for this object.
            return NotImplemented

    @override
    # pyrefly: ignore [bad-override]
    def persistent_id(self, obj: object) -> str | None:
        # [FUNCTION DOC] Return the fixed unpickle-state identifier string when obj is this
        # pickler's state token; otherwise None so the object is pickled normally.
        if obj is self._unpickle_state:
            return "unpickle_state"
        else:
            return None

    @classmethod
    def dumps(cls, obj: object, options: Options | None = None) -> bytes:
        # [FUNCTION DOC] Pickle obj with a fresh pickler into memory (with slice pytree support
        # enabled) and return the bytes. May raise errors from pickling, cache-bypass signals
        # from the op filter, or assertion errors for raw nodes.
        """
        Pickle an object.
        """
        with patch_pytree_map_over_slice(), io.BytesIO() as stream:
            pickler = cls(stream, options)
            pickler.dump(obj)
            return stream.getvalue()

    @staticmethod
    def loads(data: bytes, fake_mode: FakeTensorMode) -> object:
        # [FUNCTION DOC] Unpickle bytes produced by dumps, reconstructing tensors and shape
        # information against the given fake mode (whose existing shape env is reused and
        # overwritten). Runs under a timing context and slice pytree support. Returns the
        # object.
        """
        Unpickle an object.
        """
        from torch._dynamo.utils import dynamo_timed

        with patch_pytree_map_over_slice(), dynamo_timed("GraphPickler.loads"):
            state = _UnpickleState(fake_mode)
            with io.BytesIO(data) as stream:
                unpickler = _GraphUnpickler(stream, state)
                return unpickler.load()

    @classmethod
    def debug_dumps(
        cls,
        obj: object,
        options: "Options | None" = None,
        *,
        max_depth: int = 80,
        max_iter_items: int = 50,
        verbose: bool = True,
    ) -> str | None:
        # [FUNCTION DOC] Diagnostic: recursively search obj for the first sub-object that fails
        # to pickle and return a path string to it ("root" based), or None if everything
        # pickles. Each candidate is tested by pickling it alone; failing builtin containers,
        # bounded iterator prefixes, custom reduce arguments, dataclass fields, saved state,
        # instance attributes, slots and finally the reduce protocol are explored in that order,
        # with depth and revisit limits. Prints progress when verbose.
        # KNOWN ISSUE: the returned path can be the container itself when no child fails.
        """
        Find the first leaf that GraphPickler.dumps cannot serialize and return its path.

        This is GraphPickler-aware and avoids infinite loops by:
          - Traversing builtin containers directly (dict/list/tuple/set) instead of
            exploring their __reduce_ex__ tuples.
          - Only using __reduce_ex__ / __reduce__ for "opaque" objects.
          - Bounding recursion depth and iterator expansion.

        Args:
            obj: The object to attempt to pickle and debug.
            options: Optional Options instance for the GraphPickler.
            max_depth: Maximum recursion depth before stopping traversal.
            max_iter_items: Maximum number of items to materialize from iterators.
            verbose: If True, prints detailed traversal information.

        Returns:
            A string representing the path to the first unpicklable leaf,
            or None if the object is fully picklable.
        """
        options = options or Options()
        pickler = cls(io.BytesIO(), options)

        visited: set[int] = set()

        def log(msg: str) -> None:
            # [FUNCTION DOC] HELPER: Print the message when verbose is enabled.
            if verbose:
                print(msg)

        def fail_exc(o: Any) -> BaseException | None:
            # [FUNCTION DOC] HELPER: Try pickling o alone; return the raised exception or None.
            try:
                cls.dumps(o, options)
                return None
            except Exception as e:
                return e

        def walk(o: Any, path: str, depth: int) -> str | None:
            # [FUNCTION DOC] HELPER: Recursive search step. Returns None for already visited or
            # fine-to-pickle objects, a depth-limit marker past the depth bound, else the path of
            # the deepest-first failing part found, or this path if no child is blamed.
            if depth > max_depth:
                log(f"{'  ' * depth}Depth limit at {path} ({type(o)})")
                return path + " (depth_limit)"

            key = id(o)
            if key in visited:
                return None
            visited.add(key)

            indent = "  " * depth
            log(f"{indent}Walking: {path} ({type(o)})")

            e = fail_exc(o)
            if e is None:
                log(f"{indent}✓ Pickles fine alone")
                return None
            log(f"{indent}[FAIL pickle] {type(o)} -> {e}")

            # 1) Builtin containers: walk contents directly (do NOT call __reduce_ex__)
            if isinstance(o, dict):
                for k, v in o.items():
                    bad = walk(v, f"{path}[{k!r}]", depth + 1)
                    if bad:
                        return bad
                return path

            if isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    bad = walk(v, f"{path}[{i}]", depth + 1)
                    if bad:
                        return bad
                return path

            if isinstance(o, (set, frozenset)):
                for i, v in enumerate(o):
                    bad = walk(v, f"{path}[{i}]", depth + 1)
                    if bad:
                        return bad
                return path

            # 2) Iterator types: materialize a bounded prefix
            if hasattr(o, "__iter__") and type(o).__name__.endswith("iterator"):
                try:
                    prefix = list(itertools.islice(iter(o), max_iter_items + 1))
                except Exception:
                    prefix = None
                if prefix is not None:
                    if len(prefix) > max_iter_items:
                        log(
                            f"{indent}⚠ Iterator has more than {max_iter_items} items, "
                            f"only checking first {max_iter_items}"
                        )
                        prefix = prefix[:max_iter_items]
                    for i, v in enumerate(prefix):
                        bad = walk(v, f"{path}[{i}]", depth + 1)
                        if bad:
                            return bad
                    return path

            # 3) GraphPickler reducer_override
            try:
                red = pickler.reducer_override(o)
                log(f"{indent}reducer_override -> {type(red)}")
            except Exception as e2:
                log(f"{indent}💥 reducer_override crashed: {e2}")
                return path

            if red is not NotImplemented:
                _, args = red
                log(f"{indent}Using custom reduce, args={len(args)}")
                for i, a in enumerate(args):
                    bad = walk(a, f"{path}.reduce_args[{i}]", depth + 1)
                    if bad:
                        return bad

            # 4) Dataclasses
            if dataclasses.is_dataclass(o):
                for f in dataclasses.fields(o):
                    try:
                        v = getattr(o, f.name)
                    except Exception:
                        return f"{path}.{f.name}"
                    bad = walk(v, f"{path}.{f.name}", depth + 1)
                    if bad:
                        return bad
                return path

            # 5) __getstate__ and __dict__/__slots__
            getstate = getattr(o, "__getstate__", None)
            if callable(getstate):
                try:
                    state = getstate()
                    log(f"{indent}__getstate__ -> {type(state)}")
                except Exception as e3:
                    log(f"{indent}💥 __getstate__ failed: {e3}")
                    return path + ".__getstate__()"
                bad = walk(state, path + ".__getstate__()", depth + 1)
                if bad:
                    return bad

            if hasattr(o, "__dict__"):
                for name, v in vars(o).items():
                    bad = walk(v, f"{path}.{name}", depth + 1)
                    if bad:
                        return bad
                return path

            if hasattr(o, "__slots__"):
                for slot in o.__slots__:
                    if hasattr(o, slot):
                        bad = walk(getattr(o, slot), f"{path}.{slot}", depth + 1)
                        if bad:
                            return bad
                return path

            # 6) Last resort: reduce protocol for non-container / opaque objects
            reduce_tuple = None
            try:
                if hasattr(o, "__reduce_ex__"):
                    reduce_tuple = o.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
                    log(f"{indent}__reduce_ex__ -> {type(reduce_tuple)}")
                elif hasattr(o, "__reduce__"):
                    reduce_tuple = o.__reduce__()
                    log(f"{indent}__reduce__ -> {type(reduce_tuple)}")
            except Exception as e4:
                log(f"{indent}💥 reduce protocol failed: {e4}")
                return path

            if isinstance(reduce_tuple, tuple):
                for i, part in enumerate(reduce_tuple):
                    if part is None:
                        continue
                    bad = walk(part, f"{path}.__reduce__[{i}]", depth + 1)
                    if bad:
                        return bad

            return path

        bad = walk(obj, "root", 0)
        return bad


class _UnpickleState:
    # [CLASS DOC] HELPER: Shared context during unpickling: the fake mode to build tensors in
    # and a meta converter reused across tensors so views and duplicates map consistently.
    def __init__(self, fake_mode: FakeTensorMode) -> None:
        # [FUNCTION DOC] HELPER: Store the fake mode and create a fresh meta converter.
        self.fake_mode = fake_mode
        self.meta_converter: MetaConverter[FakeTensor] = MetaConverter()


# This token is passed when pickling to indicate that we want to use the
# unpickler's _UnpickleState as a parameter in that position.
_UnpickleStateToken = NewType("_UnpickleStateToken", object)


# pyrefly: ignore [invalid-inheritance]
class _GraphUnpickler(pickle.Unpickler):
    # [CLASS DOC] HELPER: Unpickler that resolves the persistent-id state token written by
    # GraphPickler to the unpickle state object it was constructed with.
    def __init__(self, stream: io.BytesIO, unpickle_state: _UnpickleState) -> None:
        # [FUNCTION DOC] HELPER: Read from the stream and remember the state to hand out.
        super().__init__(stream)
        self._unpickle_state = unpickle_state

    @override
    # pyrefly: ignore [bad-override]
    def persistent_load(self, pid: object) -> object:
        # [FUNCTION DOC] HELPER: Return the unpickle state for the known identifier string;
        # raise an unpickling error for any other identifier.
        if pid == "unpickle_state":
            return self._unpickle_state
        else:
            raise pickle.UnpicklingError("Invalid persistent ID")


class _ShapeEnvPickleData:
    # [CLASS DOC] HELPER: Picklable snapshot of a ShapeEnv: a copy of its attribute mapping
    # minus the tracked fakes and fake tensor cache. Unpickling copies these attributes onto
    # the shape env of the target fake mode rather than creating a new ShapeEnv.
    data: dict[str, object]

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: ShapeEnv
    ) -> tuple[
        Callable[[Self, _UnpickleState], ShapeEnv], tuple[Self, _UnpickleStateToken]
    ]:
        # [FUNCTION DOC] HELPER: Return the unpickle method plus arguments (snapshot of obj and
        # the pickler's state token) for use as a reduction.
        return cls.unpickle, (cls(obj), pickler._unpickle_state)

    def __init__(self, env: ShapeEnv) -> None:
        # [FUNCTION DOC] HELPER: Snapshot the env's attributes, dropping tracked fakes and the
        # fake tensor cache. Raises an assertion error if translation validation is enabled.
        # In theory pickle should recognize that a given ShapeEnv was already
        # pickled and reuse the resulting _ShapeEnvPickleData (so two objects
        # pointing at the same ShapeEnv get the same ShapeEnv out).
        if env._translation_validation_enabled:
            raise AssertionError("Translation validation must be disabled for pickling")
        self.data = env.__dict__.copy()
        del self.data["tracked_fakes"]
        del self.data["fake_tensor_cache"]

    def unpickle(self, unpickle_state: _UnpickleState) -> ShapeEnv:
        # [FUNCTION DOC] HELPER: Copy each saved attribute onto the existing shape env of the
        # state's fake mode (mutating it) and return that env. Raises an assertion error if
        # the fake mode or its shape env is missing.
        # Fill in the existing ShapeEnv rather than creating a new one
        if not unpickle_state.fake_mode:
            raise AssertionError("unpickle_state.fake_mode is not set")
        if not unpickle_state.fake_mode.shape_env:
            raise AssertionError("unpickle_state.fake_mode.shape_env is not set")

        for k, v in self.data.items():
            setattr(unpickle_state.fake_mode.shape_env, k, v)

        return unpickle_state.fake_mode.shape_env


class _SymNodePickleData:
    # [CLASS DOC] HELPER: Picklable form of a symbolic node: its expression, shape env, Python
    # type and hint. Unpickling rebuilds a symbolic node and wraps it as a symbolic int.
    @classmethod
    def reduce_helper(
        cls,
        pickler: GraphPickler,
        obj: _SymNodeT,
    ) -> tuple[
        Callable[[Self, _UnpickleState], _SymNodeT], tuple[Self, _UnpickleStateToken]
    ]:
        # [FUNCTION DOC] HELPER: Return the symbolic-int unpickle method plus arguments. Raises
        # NotImplementedError for any type other than a symbolic int (e.g. symbolic floats).
        args = (cls(obj.node), pickler._unpickle_state)
        if isinstance(obj, torch.SymInt):
            # pyrefly: ignore [bad-return]
            return _SymNodePickleData.unpickle_sym_int, args
        else:
            raise NotImplementedError(f"Unhandled SymNode type {type(obj)}")

    def __init__(self, node: SymNode) -> None:
        # [FUNCTION DOC] HELPER: Save the node's expression, shape env, Python type and hint.
        self.expr = node._expr
        self.shape_env = node.shape_env
        self.pytype = node.pytype
        self.hint = node._hint

    def _to_sym_node(self) -> SymNode:
        # [FUNCTION DOC] HELPER: Rebuild a symbolic node from the saved fields. Raises an
        # assertion error if no shape env was saved.
        if self.shape_env is None:
            raise AssertionError("shape_env is None")
        return SymNode(self.expr, self.shape_env, self.pytype, self.hint)

    def unpickle_sym_int(self, unpickle_state: _UnpickleState) -> torch.SymInt:
        # [FUNCTION DOC] HELPER: Return a symbolic int wrapping a rebuilt node; the state is
        # unused (the saved shape env is used directly, not the fake mode's).
        return torch.SymInt(self._to_sym_node())


class _TensorPickleData:
    # [CLASS DOC] HELPER: Picklable description of a fake tensor (a meta tensor descriptor with
    # the fake mode removed). Unpickling recreates a fake tensor in the state's fake mode via
    # its shared meta converter, restoring the fake mode on views' bases too.
    metadata: MetaTensorDesc[FakeTensor]

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: Tensor
    ) -> tuple[
        Callable[[Self, _UnpickleState], FakeTensor], tuple[Self, _UnpickleStateToken]
    ]:
        # [FUNCTION DOC] HELPER: Return the unpickle method plus arguments (description built
        # with the pickler's shared describer, and the state token).
        return cls.unpickle, (
            cls(pickler._meta_tensor_describer, obj),
            pickler._unpickle_state,
        )

    def __init__(self, describer: MetaTensorDescriber, t: Tensor) -> None:
        # [FUNCTION DOC] HELPER: Describe the tensor with the shared describer and store the
        # description without its fake mode. Raises an assertion error if the view function is
        # a custom one, or if any other unserializable field of the description is set.
        # THINGS TO WORRY ABOUT:
        # 1. Need to make sure that two tensors with the same id end up with the
        #    same id on the other side of the wire.

        metadata = describer.describe_tensor(t)

        # view_func is fine if it's either None or a _FakeTensorViewFunc. A
        # custom one (which is basically a lambda) can't be serialized.
        if metadata.view_func and not isinstance(
            metadata.view_func, torch._subclasses.meta_utils._FakeTensorViewFunc
        ):
            raise AssertionError(
                f"view_func must be None or _FakeTensorViewFunc, got "
                f"{type(metadata.view_func)}"
            )
        self.metadata = dataclasses.replace(metadata, fake_mode=None)

        # Some debugging/verification
        for k in MetaTensorDesc._UNSERIALIZABLE:
            if k in ("fake_mode", "view_func"):
                continue
            if getattr(self.metadata, k) is not None:
                raise AssertionError(f"not None: {k}: {getattr(self.metadata, k)}")

    def unpickle(self, unpickle_state: _UnpickleState) -> FakeTensor:
        # [FUNCTION DOC] HELPER: Attach the state's fake mode to the description (and to the
        # base if a view) and build a fake tensor through the shared meta converter, wrapping
        # the created meta tensor with dispatch disabled. Returns the fake tensor.
        # TODO: make common w/ _output_from_cache_entry() in fake_tensor.py?
        metadata = dataclasses.replace(
            self.metadata,
            fake_mode=unpickle_state.fake_mode,
        )

        # also need to set the fake_mode on the base of a tensor if it's a view
        if metadata.is_view and metadata.base is not None:
            new_base = dataclasses.replace(
                metadata.base,
                fake_mode=unpickle_state.fake_mode,
            )
            metadata = dataclasses.replace(metadata, base=new_base)

        def with_fake(
            make_meta_t: Callable[[], torch.Tensor], device: torch.device | str
        ) -> FakeTensor:
            # [FUNCTION DOC] HELPER: Callback for the meta converter: wrap the meta tensor made
            # by the given factory in a fake tensor of the state's fake mode on the device.
            with no_dispatch():
                return FakeTensor(
                    unpickle_state.fake_mode,
                    make_meta_t(),
                    # pyrefly: ignore [bad-argument-type]
                    device,
                )

        return unpickle_state.meta_converter.meta_tensor(
            metadata,
            unpickle_state.fake_mode.shape_env,
            with_fake,
            None,
            None,
        )


class _TorchNumpyPickleData:
    # [CLASS DOC] HELPER: Picklable reference to a numpy function by module and name, used for
    # the torch-numpy replacement functions; unpickling looks up the numpy function and maps
    # it back to the torch-numpy equivalent.
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: object
    ) -> (
        tuple[
            Callable[[Self, _UnpickleState], object], tuple[Self, _UnpickleStateToken]
        ]
        | None
    ):
        # [FUNCTION DOC] HELPER: If obj is a known torch-numpy function return the unpickle
        # method plus arguments; otherwise None.
        if data := cls.from_object(obj):
            return (cls.unpickle, (data, pickler._unpickle_state))
        else:
            return None

    def __init__(self, mod: str, name: str) -> None:
        # [FUNCTION DOC] HELPER: Store the numpy module name and function name.
        self.mod = mod
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> Callable[..., object]:
        # [FUNCTION DOC] HELPER: Import the module, fetch the numpy function by name and return
        # its torch-numpy replacement from the dynamo mapping.
        np = getattr(importlib.import_module(self.mod), self.name)
        return torch._dynamo.variables.misc.get_np_to_tnp_map()[np]

    @classmethod
    def from_object(cls, tnp: object) -> Self | None:
        # [FUNCTION DOC] HELPER: If tnp is a callable listed in the dynamo torch-numpy to numpy
        # mapping, return a reference to the numpy function (module defaults to numpy);
        # return None if it is not callable, unhashable, unmapped or nameless. Raises an
        # assertion error if the named module attribute is not the mapped numpy object.
        if not callable(tnp):
            return None

        tnp_to_np = torch._dynamo.variables.misc.get_tnp_to_np_map()
        try:
            if not (np := tnp_to_np.get(tnp)):
                return None
        except TypeError:
            return None

        if not (mod := getattr(np, "__module__", None)):
            mod = "numpy"

        if not (name := getattr(np, "__name__", None)):
            return None

        # pyrefly: ignore [unbound-name]
        if np != getattr(importlib.import_module(mod), name):
            raise AssertionError(
                f"Numpy object mismatch for {mod}.{name}"  # pyrefly: ignore [unbound-name]
            )
        # pyrefly: ignore [unbound-name]
        return cls(mod, name)


class _GraphModulePickleData:
    # [CLASS DOC] HELPER: Picklable form of a GraphModule: its attribute mapping without the
    # graph, plus a _GraphPickleData for the graph. Unpickling builds a plain GraphModule
    # (even for lazy ones) and rebuilds its graph.
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch.fx.GraphModule
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch.fx.GraphModule],
        tuple[Self, _UnpickleStateToken],
    ]:
        # [FUNCTION DOC] HELPER: Return the unpickle method plus arguments (snapshot built with
        # the pickler's options, and the state token).
        return cls.unpickle, (
            cls(obj, pickler.options),
            pickler._unpickle_state,
        )

    def __init__(self, gm: torch.fx.GraphModule, options: Options) -> None:
        # [FUNCTION DOC] HELPER: Force code generation on the module (real recompile for lazy
        # modules), copy its state without the graph and pickle-wrap the graph. Side effect:
        # recompiles the module in place.
        # Need to do this to ensure the code is created for later pickling.
        if isinstance(gm, torch.fx._lazy_graph_module._LazyGraphModule):
            _python_code = gm._real_recompile()
        else:
            _python_code = gm.recompile()
        if hasattr(gm, "__getstate__"):
            self.gm_dict = gm.__getstate__()
        else:
            self.gm_dict = gm.__dict__.copy()
        del self.gm_dict["_graph"]
        self.graph = _GraphPickleData(gm._graph, options)

    def unpickle(self, unpickle_state: _UnpickleState) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Create a bare GraphModule, install the saved state and the
        # rebuilt graph, and return it. The constructor is bypassed.
        gm = torch.fx.GraphModule.__new__(torch.fx.GraphModule)
        gm.__dict__ = self.gm_dict
        gm._graph = self.graph.unpickle(gm, unpickle_state)
        return gm


class _NodePickleData:
    # [CLASS DOC] HELPER: Picklable form of one graph node: name, op kind, pickled target, type,
    # filtered metadata, and args/kwargs in which other nodes are replaced by their pickle data.
    def __init__(
        self,
        node: torch.fx.Node,
        mapping: dict[torch.fx.Node, "_NodePickleData"],
        options: Options,
    ) -> None:
        # [FUNCTION DOC] HELPER: Capture the node. Args and kwargs nodes are looked up in the
        # mapping (so inputs must have been captured earlier; a missing one raises a key
        # error). The target is pickled through the op helper (may raise for unsupported
        # targets), and metadata keys are kept only if they pass the filter.
        self.args = pytree.tree_map_only(torch.fx.Node, lambda n: mapping[n], node.args)
        self.kwargs = pytree.tree_map_only(
            torch.fx.Node, lambda n: mapping[n], node.kwargs
        )
        # -- self.graph = node.graph
        self.name = node.name
        self.op = node.op
        self.target = _OpPickleData.pickle(node.target, options)
        # self.input_nodes = node._input_nodes
        # self.users = node.users
        self.type = node.type
        # self.sort_key = node._sort_key
        # self.repr_fn = node._repr_fn
        # self.meta = node.meta
        self.meta = {
            k: v
            for k, v in node.meta.items()
            if (
                not options.node_metadata_key_filter
                or options.node_metadata_key_filter(k)
            )
        }

    def unpickle(
        self,
        graph: torch.fx.Graph,
        mapping: dict["_NodePickleData", torch.fx.Node],
        unpickle_state: _UnpickleState,
    ) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Recreate the node inside the given graph, mapping argument
        # pickle data back to already created nodes, resolving the target, and attaching the
        # saved metadata. Raises an assertion error if the target is neither callable nor a
        # string. Returns the new node.
        args = pytree.tree_map_only(_NodePickleData, lambda n: mapping[n], self.args)
        kwargs = pytree.tree_map_only(
            _NodePickleData, lambda n: mapping[n], self.kwargs
        )
        target = self.target.unpickle(unpickle_state)
        if not (callable(target) or isinstance(target, str)):
            raise AssertionError(f"target must be callable or str, got {type(target)}")
        node = graph.create_node(self.op, target, args, kwargs, self.name, self.type)
        node.meta = self.meta
        return node


class _OpPickleData:
    # [CLASS DOC] HELPER: Abstract base for picklable references to graph-node targets (strings,
    # operator overloads, packets, precompiled artifacts, standard functions). The pickle
    # factory picks the subclass; each subclass implements unpickle to resolve the target.
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, op: object
    ) -> tuple[Callable[[_UnpickleState], object], tuple[_UnpickleStateToken]]:
        # [FUNCTION DOC] HELPER: Pickle the op into a helper and return its unpickle method with
        # the state token as the argument.
        result = cls.pickle(op, pickler.options)
        return (result.unpickle, (pickler._unpickle_state,))

    @classmethod
    def pickle(cls, op: object, options: Options) -> "_OpPickleData":
        # [FUNCTION DOC] HELPER: Choose and build the helper for a target: strings kept as is;
        # objects wrapping a precompiled artifact serialized; operator overloads and packets
        # go through the op filter; functions whose pretty name starts with a supported root
        # (builtins, math, torch, operator, einops) split into root and detail. Anything else
        # raises NotImplementedError.
        # KNOWN ISSUE: the redundant attribute-existence check after the wrapped-artifact test
        # can never fail; the TODO notes unsupported targets should bypass the cache instead.
        if isinstance(op, str):
            return _OpStrPickleData(op)

        if isinstance(getattr(op, "__wrapped__", None), AOTCompiledArtifact):
            if not hasattr(op, "__wrapped__"):
                raise AssertionError("op missing __wrapped__ attribute")
            artifact = op.__wrapped__
            if not isinstance(artifact, AOTCompiledArtifact):
                raise AssertionError(
                    f"Expected AOTCompiledArtifact, got {type(artifact)}"
                )
            return _OpPrecompiledPickleData(artifact)

        name = torch.fx.Node._pretty_print_target(op)

        if isinstance(op, torch._ops.OpOverload):
            return cls._pickle_op(name, _OpOverloadPickleData, options)
        elif isinstance(op, torch._ops.OpOverloadPacket):
            return cls._pickle_op(name, _OpOverloadPacketPickleData, options)
        elif name.startswith(_OpFunctionPickleData.SUPPORTED_ROOTS):
            root, detail = name.split(".", 1)
            return _OpFunctionPickleData(root, detail)
        else:
            # TODO: raise a BypassFxGraphCache so we will just bypass this one...
            raise NotImplementedError(f"TARGET: {type(op)} {op} {name}")

    @staticmethod
    def _pickle_op(
        name: str,
        datacls: type["_OpOverloadPickleData"] | type["_OpOverloadPacketPickleData"],
        options: Options,
    ) -> "_OpPickleData":
        # [FUNCTION DOC] HELPER: If an ops filter exists and rejects the name, signal a cache
        # bypass through the inductor cacheability validator (which may raise). Otherwise
        # return a new instance of the given helper class for the name.
        if (ops_filter := options.ops_filter) and not ops_filter(name):
            from torch._inductor.codecache import CacheabilityValidator

            CacheabilityValidator.bypass(f"Unable to pickle non-standard op: {name}")
        return datacls(name)

    @abstractmethod
    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        # [FUNCTION DOC] HELPER: Abstract hook; subclasses return the real target object.
        pass

    @classmethod
    def _lookup_global_by_name(cls, name: str) -> object:
        # [FUNCTION DOC] HELPER: Resolve a possibly dotted name starting from this module's global
        # names (the first part must be a global here, e.g. torch), then attribute lookups for
        # the rest. Raises a key error or attribute error if not found.
        """
        Like `globals()[name]` but supports dotted names.
        """
        if "." in name:
            mod, rest = name.split(".", 1)
            root = globals()[mod]
            return cls._getattr_by_name(root, rest)
        else:
            return globals()[name]

    @staticmethod
    def _getattr_by_name(root: object, name: str) -> object:
        # [FUNCTION DOC] HELPER: Follow a dotted attribute path starting at root and return the
        # final attribute. Raises an attribute error if any part is missing.
        """
        Like `getattr(root, name)` but supports dotted names.
        """
        while "." in name:
            mod, name = name.split(".", 1)
            root = getattr(root, mod)
        return getattr(root, name)


class _OpStrPickleData(_OpPickleData):
    # [CLASS DOC] HELPER: Target that is just a string name (e.g. a method or attribute name).
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Store the string.
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> str:
        # [FUNCTION DOC] HELPER: Return the stored string.
        return self.name


class _OpOverloadPickleData(_OpPickleData):
    # [CLASS DOC] HELPER: Reference to an operator overload by its dotted name.
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Store the dotted name.
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverload:
        # [FUNCTION DOC] HELPER: Look up the name and return the operator overload. Raises an
        # assertion error if the result is not an operator overload.
        obj = self._lookup_global_by_name(self.name)
        if not isinstance(obj, torch._ops.OpOverload):
            raise AssertionError(f"Expected OpOverload, got {type(obj)}")
        return obj


class _OpOverloadPacketPickleData(_OpPickleData):
    # [CLASS DOC] HELPER: Reference to an operator overload packet by its dotted name.
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Store the dotted name.
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverloadPacket:
        # [FUNCTION DOC] HELPER: Look up the name and return the overload packet. Raises an
        # assertion error if the result is not an overload packet.
        obj = self._lookup_global_by_name(self.name)
        if not isinstance(obj, torch._ops.OpOverloadPacket):
            raise AssertionError(f"Expected OpOverloadPacket, got {type(obj)}")
        return obj


class _OpPrecompiledPickleData(_OpPickleData):
    # [CLASS DOC] HELPER: Target that is a precompiled artifact, stored as its serialized bytes.
    def __init__(self, artifact: AOTCompiledArtifact) -> None:
        # [FUNCTION DOC] HELPER: Serialize the artifact and keep the bytes.
        self.contents = artifact.serialize()

    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        # [FUNCTION DOC] HELPER: Deserialize the artifact and return a wrapper function that
        # forwards positional arguments to it (keyword arguments are not supported).
        precompiled_artifact = AOTCompiledArtifact.deserialize(self.contents)
        import functools

        @functools.wraps(precompiled_artifact)
        def wrapped(*args: Any) -> Any:
            # [FUNCTION DOC] HELPER: Call the deserialized artifact with the given arguments.
            return precompiled_artifact(*args)

        return wrapped


class _OpFunctionPickleData(_OpPickleData):
    """
    Supports pickling a set of standard/common functions
    These must be prefixed with the full namespace in order to properly
    be pickled (i.e `einops.rearrange` and not `from einops import rearrange`)
    """
    # [CLASS DOC] HELPER: Target that is a standard function identified by a root module name
    # (one of the supported roots) and a dotted detail name; unpickling imports the root and
    # follows the detail path.

    # Static variable listing supported root names
    SUPPORTED_ROOTS = ("builtins.", "math.", "torch.", "operator.", "einops.")

    def __init__(self, root: str, name: str) -> None:
        # [FUNCTION DOC] HELPER: Store the root module name and the remaining dotted name.
        self.root = root
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        # [FUNCTION DOC] HELPER: Resolve the function from its root module (builtins via the
        # builtins mapping, which yields None for unknown names; others by attribute path).
        # Raises NotImplementedError for an unsupported root.
        if self.root == "builtins":
            return __builtins__.get(self.name)  # type: ignore[attr-defined]
        elif self.root == "math":
            import math

            return self._getattr_by_name(math, self.name)
        elif self.root == "torch":
            return self._getattr_by_name(torch, self.name)
        elif self.root == "operator":
            import operator

            return self._getattr_by_name(operator, self.name)
        elif self.root == "einops":
            import einops

            return self._getattr_by_name(einops, self.name)
        else:
            raise NotImplementedError


class _GraphPickleData:
    # [CLASS DOC] HELPER: Picklable form of a Graph: tracer class and extras, the ordered tuple of
    # node pickle data, and the code generator. Other graph state is rebuilt on unpickle.
    def __init__(self, graph: torch.fx.Graph, options: Options) -> None:
        # [FUNCTION DOC] HELPER: Capture each node in order (earlier nodes available to later
        # ones), the tracer class and extras, and the code generator.
        self.tracer_cls = graph._tracer_cls
        self.tracer_extras = graph._tracer_extras

        nodes: dict[torch.fx.Node, _NodePickleData] = {}
        for node in graph.nodes:
            nodes[node] = _NodePickleData(node, nodes, options)
        self.nodes = tuple(nodes.values())
        self._codegen = graph._codegen

        # Unpickled variables:
        # self._used_names = graph._used_names
        # -- self._insert = self._root.prepend
        # self._len = graph._len
        # self._graph_namespace = graph._graph_namespace
        # self._owning_module = graph._owning_module
        # self._co_fields: Dict[str, Any] = graph._co_fields
        # -- self._find_nodes_lookup_table = _FindNodesLookupTable()

    def unpickle(
        self, gm: torch.fx.GraphModule, unpickle_state: _UnpickleState
    ) -> torch.fx.Graph:
        # [FUNCTION DOC] HELPER: Create a new graph owned by gm, recreate nodes in saved order
        # and restore the code generator. Returns the graph.
        # KNOWN ISSUE: the attribute-existence check for the code generator is always true here.
        graph = torch.fx.Graph(gm, self.tracer_cls, self.tracer_extras)

        nodes: dict[_NodePickleData, torch.fx.Node] = {}
        for nd in self.nodes:
            nodes[nd] = nd.unpickle(graph, nodes, unpickle_state)
        if hasattr(self, "_codegen"):
            graph._codegen = self._codegen

        return graph


class _TracingContextPickleData:
    # [CLASS DOC] HELPER: Picklable subset of a TracingContext (module context, frame info, graph
    # name, flattened parameter info, output strides, unspecialization flag). Other fields
    # are deliberately not saved. Unpickling builds a new context on the state's fake mode.
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch._guards.TracingContext
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch._guards.TracingContext],
        tuple[Self, _UnpickleStateToken],
    ]:
        # [FUNCTION DOC] HELPER: Return the unpickle method plus arguments (snapshot of obj and
        # the state token).
        return (
            cls.unpickle,
            (
                cls(obj),
                pickler._unpickle_state,
            ),
        )

    def __init__(self, context: TracingContext) -> None:
        # [FUNCTION DOC] HELPER: Copy the selected fields from the context by reference.
        # TODO: Do we really need all of this?
        self.module_context = context.module_context
        self.frame_summary_stack = context.frame_summary_stack
        self.loc_in_frame = context.loc_in_frame
        self.aot_graph_name = context.aot_graph_name
        self.params_flat = context.params_flat
        self.params_flat_unwrap_subclasses = context.params_flat_unwrap_subclasses
        self.params_unwrapped_to_flat_index = context.params_unwrapped_to_flat_index
        self.output_strides = context.output_strides
        self.force_unspec_int_unbacked_size_like = (
            context.force_unspec_int_unbacked_size_like
        )
        # Not saved (because it's difficult and maybe not needed?):
        #   self.fw_metadata = context.fw_metadata
        #   self.guards_context = None
        #   self.global_context = None
        #   self.fake_mode = None
        #   self.fakify_first_call = None
        #   self.hop_dispatch_set_cache = None
        #   self.tensor_to_context = context.tensor_to_context

    def unpickle(self, unpickle_state: _UnpickleState) -> TracingContext:
        # [FUNCTION DOC] HELPER: Create a tracing context on the state's fake mode and fill in
        # the saved fields; returns it.
        context = TracingContext(unpickle_state.fake_mode)
        context.module_context = self.module_context
        context.frame_summary_stack = self.frame_summary_stack
        context.loc_in_frame = self.loc_in_frame
        context.aot_graph_name = self.aot_graph_name
        context.params_flat = self.params_flat
        context.params_flat_unwrap_subclasses = self.params_flat_unwrap_subclasses
        context.params_unwrapped_to_flat_index = self.params_unwrapped_to_flat_index
        context.output_strides = self.output_strides
        context.force_unspec_int_unbacked_size_like = (
            self.force_unspec_int_unbacked_size_like
        )
        return context
