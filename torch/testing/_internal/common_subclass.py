# mypy: ignore-errors

# [MODULE DOC]
# Example tensor subclasses used in tests: wrapper subclasses (custom sizes, custom strides,
# diagonal-backed, sparse-like), a non-wrapper subclass carrying extra state, a subclass that
# records torch function calls, and one with a tensor-subclass flatten/unflatten protocol.
# Also exposes subclass_db, a mapping from tensor class to SubclassInfo describing how to
# create an instance of each. Import has no side effects beyond building these definitions.

import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
import torch.utils._pytree as pytree


# TODO: Move LoggingTensor here.
from torch.testing._internal.logging_tensor import LoggingTensor


# Base class for wrapper-style tensors.
class WrapperTensor(torch.Tensor):
    # [CLASS DOC] Base class for wrapper-style tensor subclasses that hold no storage of their
    # own. Subclasses implement get_wrapper_properties to return an example tensor plus keyword
    # overrides (size, dtype, layout, device, requires_grad); construction then creates a
    # wrapper subclass with those properties and checks that metadata properties are not
    # overridden in Python.
    @staticmethod
    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC] Creates the wrapper tensor. Calls get_wrapper_properties, then fills in
        # missing size, dtype, layout and device from the example tensor (requires_grad defaults
        # to false) and builds a wrapper subclass instance, which is validated before return.
        # Raises RuntimeError (via validation) if a forbidden property is overridden.
        t, kwargs = cls.get_wrapper_properties(*args, **kwargs)
        if "size" not in kwargs:
            size = t.size()
        else:
            size = kwargs["size"]
            del kwargs["size"]
        if "dtype" not in kwargs:
            kwargs["dtype"] = t.dtype
        if "layout" not in kwargs:
            kwargs["layout"] = t.layout
        if "device" not in kwargs:
            kwargs["device"] = t.device
        if "requires_grad" not in kwargs:
            kwargs["requires_grad"] = False
        # Ignore memory_format and pin memory for now as I don't know how to
        # safely access them on a Tensor (if possible??)

        wrapper = torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)
        wrapper._validate_methods()
        return wrapper

    @classmethod
    def get_wrapper_properties(cls, *args, **kwargs):
        # [FUNCTION DOC] Abstract hook: subclasses return a pair of an example tensor and a
        # mapping of keyword overrides for the wrapper. This base version always raises
        # NotImplementedError.
        # Should return both an example Tensor and a dictionary of kwargs
        # to override any of that example Tensor's properly.
        # This is very similar to the `t.new_*(args)` API
        raise NotImplementedError("You need to implement get_wrapper_properties")

    def _validate_methods(self):
        # [FUNCTION DOC] HELPER: Raises RuntimeError if the subclass overrides any of size,
        # stride, dtype, layout, device or requires_grad, since Python overrides would not be
        # seen by C++ callers. Compares class attributes against those of the base tensor class.
        # Skip this if not in debug mode?
        # Changing these on the python side is wrong as it would not be properly reflected
        # on the c++ side
        # This doesn't catch attributes set in the __init__
        forbidden_overrides = ["size", "stride", "dtype", "layout", "device", "requires_grad"]
        for el in forbidden_overrides:
            if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
                raise RuntimeError(f"Subclass {self.__class__.__name__} is overwriting the "
                                   f"property {el} but this is not allowed as such change would "
                                   "not be reflected to c++ callers.")


class WrapperTensorWithCustomSizes(WrapperTensor):
    # [CLASS DOC] Wrapper subclass around a single inner tensor stored in attribute t. Requests
    # that size queries be dispatched to Python. Operations unwrap arguments, run on the inner
    # tensors, and rewrap every tensor result in this class.
    @classmethod
    def get_wrapper_properties(cls, t, requires_grad=False):
        # [FUNCTION DOC] Returns the inner tensor t and overrides setting requires_grad and a
        # dispatch policy that routes size queries through torch dispatch.
        return t, {"requires_grad": requires_grad, "dispatch_sizes_strides_policy": "sizes"}

    def __init__(self, t, requires_grad=False):
        # [FUNCTION DOC] Stores the inner tensor t as attribute t; requires_grad is consumed
        # by construction and not stored here.
        self.t = t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Dispatch handler: returns NotImplemented if any participating type is
        # not a base of this class. Otherwise unwraps all instances in args and kwargs to their
        # inner tensors, calls func, and wraps each tensor in the (nested) result back into
        # this class. Returns the rewrapped result.
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        def unwrap(e):
            # [FUNCTION DOC] HELPER: Returns the inner tensor of an instance, else e unchanged.
            return e.t if isinstance(e, WrapperTensorWithCustomSizes) else e

        def wrap(e):
            # [FUNCTION DOC] HELPER: Wraps any tensor in this class, else returns e unchanged.
            return WrapperTensorWithCustomSizes(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    def __repr__(self):
        # [FUNCTION DOC] Returns the tensor repr using the text of the inner tensor as contents.
        return super().__repr__(tensor_contents=f"t={self.t}")


class WrapperTensorWithCustomStrides(WrapperTensor):
    # [CLASS DOC] Same as the custom sizes wrapper but requests that stride queries be
    # dispatched to Python. Holds the inner tensor in attribute t; operations unwrap, run
    # and rewrap tensor results.
    @classmethod
    def get_wrapper_properties(cls, t, requires_grad=False):
        # [FUNCTION DOC] Returns the inner tensor t and overrides setting requires_grad and a
        # dispatch policy that routes stride queries through torch dispatch.
        return t, {"requires_grad": requires_grad, "dispatch_sizes_strides_policy": "strides"}

    def __init__(self, t, requires_grad=False):
        # [FUNCTION DOC] Stores the inner tensor t as attribute t; requires_grad is not stored.
        self.t = t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Dispatch handler: returns NotImplemented if any participating type is
        # not a base of this class. Otherwise unwraps instances to inner tensors, calls func,
        # and wraps every tensor in the result back into this class.
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        def unwrap(e):
            # [FUNCTION DOC] HELPER: Returns the inner tensor of an instance, else e unchanged.
            return e.t if isinstance(e, WrapperTensorWithCustomStrides) else e

        def wrap(e):
            # [FUNCTION DOC] HELPER: Wraps any tensor in this class, else returns e unchanged.
            return WrapperTensorWithCustomStrides(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    def __repr__(self):
        # [FUNCTION DOC] Returns the tensor repr using the text of the inner tensor as contents.
        return super().__repr__(tensor_contents=f"t={self.t}")


class DiagTensorBelow(WrapperTensor):
    # [CLASS DOC] Wrapper representing a square diagonal matrix by storing only its 1D diagonal
    # in attribute diag; the logical size is the diagonal size twice. handled_ops maps op names
    # to custom handlers (empty by default); other ops fall back to densifying, running the op
    # and re-compressing results that are 1D or diagonal 2D.
    @classmethod
    def get_wrapper_properties(cls, diag, requires_grad=False):
        # [FUNCTION DOC] Returns the diagonal tensor and overrides giving the logical square
        # size (the diagonal size repeated twice) and requires_grad. Raises AssertionError if
        # diag is not 1D.
        if diag.ndim != 1:
            raise AssertionError(f"Expected diag.ndim == 1, got {diag.ndim}")
        return diag, {"size": diag.size() + diag.size(), "requires_grad": requires_grad}

    def __init__(self, diag, requires_grad=False):
        # [FUNCTION DOC] Stores the 1D diagonal as attribute diag; requires_grad is not stored.
        self.diag = diag

    handled_ops = {}

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Dispatch handler: returns NotImplemented if a participating type is not
        # a base of this class. If handled_ops has an entry for the op name it calls that
        # handler with the raw arguments. Otherwise densifies instances, runs the op, and
        # re-wraps 1D results and 2D results whose nonzeros lie on the diagonal.
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        # For everything else, call the handler:
        fn = cls.handled_ops.get(func.__name__, None)
        if fn:
            return fn(*args, **(kwargs or {}))
        else:
            # Note that here, because we don't need to provide the autograd formulas
            # we can have a default "fallback" that creates a plain Tensor based
            # on the diag elements and calls the func again.

            def unwrap(e):
                # [FUNCTION DOC] HELPER: Expands an instance into a dense diagonal matrix,
                # else returns e unchanged.
                return e.diag.diag() if isinstance(e, DiagTensorBelow) else e

            def wrap(e):
                # [FUNCTION DOC] HELPER: Wraps 1D tensors as diagonals, and 2D tensors whose
                # nonzero count equals that of their diagonal as diagonal tensors of that
                # diagonal. Anything else is returned unchanged.
                if isinstance(e, torch.Tensor) and e.ndim == 1:
                    return DiagTensorBelow(e)
                if isinstance(e, torch.Tensor) and e.ndim == 2 and e.count_nonzero() == e.diag().count_nonzero():
                    return DiagTensorBelow(e.diag())
                return e

            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
            return rs

    def __repr__(self):
        # [FUNCTION DOC] Returns the tensor repr using the text of the diagonal as contents.
        return super().__repr__(tensor_contents=f"diag={self.diag}")


class SparseTensor(WrapperTensor):
    # [CLASS DOC] Simple COO-like wrapper storing nonzero values and their indices (rows of
    # coordinates) in attributes values and indices, with a logical dense size. Ops default to
    # densifying, running, and re-sparsifying by nonzero entries; _SPECIAL_IMPLS can supply
    # per-op overrides keyed by qualified op name.
    @classmethod
    def get_wrapper_properties(cls, size, values, indices, requires_grad=False):
        # [FUNCTION DOC] Returns the values tensor with overrides giving the logical size and
        # requires_grad. Raises AssertionError if values and indices are on different devices.
        if values.device != indices.device:
            raise AssertionError(f"Expected values.device == indices.device, got {values.device} != {indices.device}")
        return values, {"size": size, "requires_grad": requires_grad}

    def __init__(self, size, values, indices, requires_grad=False):
        # [FUNCTION DOC] Stores values and indices as attributes; size and requires_grad are
        # consumed at construction and not stored.
        self.values = values
        self.indices = indices

    def __repr__(self):
        # [FUNCTION DOC] Returns the tensor repr using the values and indices as contents.
        return super().__repr__(tensor_contents=f"values={self.values}, indices={self.indices}")

    def sparse_to_dense(self):
        # [FUNCTION DOC] Builds a dense zero tensor of the logical size (default device) with
        # the values scattered at the stored indices. Returns the dense tensor.
        res = torch.zeros(self.size(), dtype=self.values.dtype)
        res[self.indices.unbind(1)] = self.values
        return res

    @staticmethod
    def from_dense(t):
        # [FUNCTION DOC] Builds a SparseTensor from a dense tensor t by recording the indices
        # and values of its nonzero entries, with the logical size of t. Returns the new tensor.
        indices = t.nonzero()
        values = t[indices.unbind(1)]
        return SparseTensor(t.size(), values, indices)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Dispatch handler: first tries a special implementation registered
        # under the qualified op name and returns its result if any. Otherwise densifies sparse
        # arguments, runs the op, and converts every tensor result back to a SparseTensor.
        func_name = f"{func.__module__}.{func.__name__}"

        res = cls._try_call_special_impl(func_name, args, kwargs)
        if res is not NotImplemented:
            return res

        # Otherwise, use a default implementation that construct dense
        # tensors and use that to compute values
        def unwrap(e):
            # [FUNCTION DOC] HELPER: Densifies a SparseTensor, else returns e unchanged.
            return e.sparse_to_dense() if isinstance(e, SparseTensor) else e

        # Wrap back all Tensors into our custom class
        def wrap(e):
            # [FUNCTION DOC] HELPER: Converts any tensor to a SparseTensor by nonzero entries,
            # else returns e unchanged.
            # Check for zeros and use that to get indices
            return SparseTensor.from_dense(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs


    _SPECIAL_IMPLS = {}

    @classmethod
    def _try_call_special_impl(cls, func, args, kwargs):
        # [FUNCTION DOC] HELPER: Looks up func (a qualified op name) in the special
        # implementation mapping and calls it with args and kwargs; returns NotImplemented if
        # there is no entry.
        if func not in cls._SPECIAL_IMPLS:
            return NotImplemented
        return cls._SPECIAL_IMPLS[func](args, kwargs)


# Example non-wrapper subclass that stores extra state.
class NonWrapperTensor(torch.Tensor):
    # [CLASS DOC] Non-wrapper subclass sharing storage with its data and carrying a Python
    # attribute extra_state (a mapping with key last_func_called) that records the name of the
    # last torch function applied to produce each instance. Deepcopy copies the state.
    def __new__(cls, data):
        # [FUNCTION DOC] Creates a subclass instance from data and initializes extra_state with
        # no last function. Returns the new tensor.
        t = torch.Tensor._make_subclass(cls, data)
        t.extra_state = {
            'last_func_called': None
        }
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Runs the default torch function handling; if the result is an instance
        # of this class, sets its extra_state: for deepcopy a deep copy of the source state,
        # otherwise a new mapping recording the function name. Returns the result.
        result = super().__torch_function__(func, types, args, kwargs)

        if isinstance(result, cls):
            # Do something with the extra state. For the example here, just store the name of the
            # last function called (skip for deepcopy so the copy has the same extra state).
            if func is torch.Tensor.__deepcopy__:
                result.extra_state = deepcopy(args[0].extra_state)
            else:
                result.extra_state = {
                    'last_func_called': func.__name__,
                }

        return result

    # new_empty() must be defined for deepcopy to work
    def new_empty(self, shape):
        # [FUNCTION DOC] Returns a new uninitialized instance of this class with the given
        # shape (default dtype); needed so deepcopy works.
        return type(self)(torch.empty(shape))


class RedispatchTensor(torch.Tensor):
    # [CLASS DOC] Non-wrapper subclass that logs torch function calls into a shared list held
    # in the slot call_log. Results of operations are rewrapped in this class and share the
    # same log list, so one list accumulates (qualified function name, types) entries.
    __slots__ = ['call_log']

    def __new__(cls, data, call_log=None):
        # [FUNCTION DOC] Creates an instance from data, preserving its requires_grad, and
        # stores call_log (a new empty list if none given). Returns the new tensor.
        t = torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)
        t.call_log = call_log if call_log is not None else []
        return t

    def __repr__(self):
        # [FUNCTION DOC] Returns the default tensor repr computed with torch function handling
        # disabled so the repr itself is not logged.
        with torch._C.DisableTorchFunction():
            return super().__repr__()

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        # [FUNCTION DOC] Finds the call log from the first instance in args (or kwargs), appends
        # an entry of function qualified name and types, redispatches the call, and wraps
        # plain tensor results in this class sharing the same log. Returns the rewrapped result.
        # Side effect: mutates the shared log list.
        call_log_entry = (func.__qualname__, types)

        call_log = None

        def find_log(x):
            # [FUNCTION DOC] HELPER: Records the log of the first instance seen into the
            # enclosing call_log variable.
            nonlocal call_log
            if isinstance(x, RedispatchTensor) and call_log is None:
                call_log = x.call_log

        _ = tree_map(find_log, args)
        if kwargs and call_log is None:
            _ = tree_map(find_log, kwargs)

        if call_log is not None:
            call_log.append(call_log_entry)

        ret = torch.overrides.redispatch_function(func, types, args, kwargs)

        def wrap(x):
            # [FUNCTION DOC] HELPER: Wraps a plain tensor in this class with the shared log,
            # else returns x unchanged.
            if isinstance(x, torch.Tensor) and not isinstance(x, RedispatchTensor):
                return cls(x, call_log=call_log)
            return x
        return tree_map(wrap, ret)


# Class used to store info about subclass tensors used in testing.
class SubclassInfo:
    # [CLASS DOC] HELPER: Plain record describing a test subclass: name (string label),
    # create_fn (callable taking a shape and returning an instance), and closed_under_ops
    # (whether ops on instances keep returning the same subclass).

    __slots__ = ['name', 'create_fn', 'closed_under_ops']

    def __init__(self, name, create_fn, closed_under_ops=True):
        # [FUNCTION DOC] HELPER: Stores the three fields as attributes.
        self.name = name
        self.create_fn = create_fn  # create_fn(shape) -> tensor instance
        self.closed_under_ops = closed_under_ops


# Helper function to create a subclass of the given class and possibly cache sizes / strides.
def _create_and_access_shape(cls, shape):
    # [FUNCTION DOC] HELPER: Creates cls around a random tensor of the given shape and queries
    # size and stride once to populate cached metadata. Returns the instance.
    sub = cls(torch.randn(shape))
    # NB: Wrapper subclasses with custom dispatched sizes / strides cache this info
    # on the first call via non-serializable PyCapsules. We purposefully trigger cache
    # population here for serialization / deepcopy tests to verify that the presence of this
    # cache info doesn't cause problems.
    sub.size()
    sub.stride()
    return sub


subclass_db = {
    torch.Tensor: SubclassInfo(
        'base_tensor', create_fn=torch.randn
    ),
    NonWrapperTensor: SubclassInfo(
        'non_wrapper_tensor',
        create_fn=lambda shape: NonWrapperTensor(torch.randn(shape))
    ),
    LoggingTensor: SubclassInfo(
        'logging_tensor',
        create_fn=lambda shape: LoggingTensor(torch.randn(shape))
    ),
    SparseTensor: SubclassInfo(
        'sparse_tensor',
        create_fn=lambda shape: SparseTensor.from_dense(torch.randn(shape).relu())
    ),
    DiagTensorBelow: SubclassInfo(
        'diag_tensor_below',
        create_fn=lambda shape: DiagTensorBelow(torch.randn(shape)),
        closed_under_ops=False  # sparse semantics
    ),
    WrapperTensorWithCustomSizes: SubclassInfo(
        'wrapper_with_custom_sizes',
        create_fn=lambda shape: _create_and_access_shape(WrapperTensorWithCustomSizes, shape),
        closed_under_ops=False,
    ),
    WrapperTensorWithCustomStrides: SubclassInfo(
        'wrapper_with_custom_strides',
        create_fn=lambda shape: _create_and_access_shape(WrapperTensorWithCustomStrides, shape),
        closed_under_ops=False,
    ),
    RedispatchTensor: SubclassInfo(
        "redispatch_tensor",
        create_fn=lambda shape: RedispatchTensor(torch.randn(shape))
    ),
}

class SubclassWithTensorFactory(torch.Tensor):
    # [CLASS DOC] Wrapper subclass around an inner tensor in attribute src, mirroring its shape,
    # strides, offset, device, layout, dtype and requires_grad. Supports the flatten/unflatten
    # protocol with the single inner tensor src. Ops unwrap (float32 inner tensors are
    # re-materialized via multiplication by ones), run, and rewrap tensor outputs.
    @staticmethod
    def __new__(cls, src):
        # [FUNCTION DOC] Creates the wrapper with the metadata of src. Returns the wrapper.
        shape = src.shape
        kwargs = {}
        kwargs["strides"] = src.stride()
        kwargs["storage_offset"] = src.storage_offset()
        kwargs["device"] = src.device
        kwargs["layout"] = src.layout
        kwargs["requires_grad"] = src.requires_grad
        kwargs["dtype"] = src.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, src):
        # [FUNCTION DOC] Stores the inner tensor as attribute src.
        self.src = src

    def __repr__(self):
        # [FUNCTION DOC] Returns just the class name.
        return f"{self.__class__.__name__}"

    def __tensor_flatten__(self):
        # [FUNCTION DOC] Returns the list of inner tensor attribute names (just src) and no
        # extra metadata.
        return ["src"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] Rebuilds an instance from the mapping of inner tensors; meta and the
        # outer size and stride are ignored (derived from src).
        src = inner_tensors["src"]
        return cls(src)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # [FUNCTION DOC] Unwraps all instances in args and kwargs via the inner tensor, calls
        # func, then wraps every tensor in the flattened output in this class and restores the
        # original output structure. Returns that structure.
        if kwargs is None:
            kwargs = {}

        def _fn(x):
            # [FUNCTION DOC] HELPER: For float32 inner tensors returns a fresh product of src
            # and ones (a new tensor); otherwise returns src itself.
            return x.src * torch.ones(x.src.shape) if x.src.dtype == torch.float32 else x.src

        _args = pytree.tree_map_only(cls, _fn, args)
        _kwargs = pytree.tree_map_only(cls, _fn, kwargs)

        _out = func(*_args, **_kwargs)

        _out_flat, _out_spec = pytree.tree_flatten(_out)

        out_flat = [cls(o) if isinstance(o, torch.Tensor) else o for o in _out_flat]
        return pytree.tree_unflatten(out_flat, _out_spec)
