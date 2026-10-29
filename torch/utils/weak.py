# mypy: allow-untyped-defs
# [MODULE DOC] Weak-reference utilities that compare keys by identity, so tensors (whose
# equality returns a tensor) can be weakly referenced and used as mapping keys.
# Exports TensorWeakRef, WeakIdRef, WeakIdKeyDictionary and its alias WeakTensorKeyDictionary.
# Also defines private helpers _IterationGuard and _WeakHashRef, and the alias WeakRef.
# No import-time side effects beyond imports and definitions.
from __future__ import annotations

import collections.abc as _collections_abc
import weakref
from collections.abc import Mapping, MutableMapping
from weakref import ref

from torch import Tensor


WeakRef = ref


__all__ = [
    "TensorWeakRef",
    "WeakIdRef",
    "WeakIdKeyDictionary",
    "WeakTensorKeyDictionary",
]


# TODO: make weakref properly thread safe following
# https://github.com/python/cpython/pull/125325
class _IterationGuard:
    # [CLASS DOC] HELPER: Context manager that marks a weak container as being iterated, so
    # that dead-key removals are queued instead of applied. Holds only a weak reference to
    # the container. Leaving the last active guard flushes the queued removals.
    # This context manager registers itself in the current iterators of the
    # weak container, such as to delay all removals until the context manager
    # exits.
    # This technique should be relatively thread-safe (since sets are).

    def __init__(self, weakcontainer) -> None:
        # [FUNCTION DOC] HELPER: Remember a weak reference to the container (avoids cycles).
        # Don't create cycles
        self.weakcontainer = ref(weakcontainer)

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Register this guard in the container's active-iteration set,
        # if the container is still alive. Returns the guard.
        w = self.weakcontainer()
        if w is not None:
            w._iterating.add(self)
        return self

    def __exit__(self, e, t, b):
        # [FUNCTION DOC] HELPER: Unregister this guard; when no guards remain, apply the
        # container's pending removals. Does not suppress exceptions.
        w = self.weakcontainer()
        if w is not None:
            s = w._iterating
            s.remove(self)
            if not s:
                w._commit_removals()


# This file defines a variant of WeakKeyDictionary that overrides the hashing
# behavior of the key to use object identity, rather than the builtin
# __eq__/__hash__ functions.  This is useful for Tensor weak keys, as their
# __eq__ implementation return a Tensor (elementwise equality), which means
# you can't use them directly with the WeakKeyDictionary in standard library.
#
# Our implementation strategy is to create a wrapper weak key object, which we
# use as a key in a stock Python dictionary.  This is similar to how weakref
# implements WeakKeyDictionary, but instead of using weakref.ref as the
# wrapper, we use a custom wrapper that has different __eq__ and __hash__
# behavior.  Note that we subsequently store this weak key directly in an
# ORDINARY dictionary, since the newly constructed WeakIdKey's only use would
# be a dictionary so it would have no strong references.  Ensuring that
# only live WeakIdKeys are in the map is handled by putting finalizers on the
# original key object.


# It is simpler to implement this with composition, but if we want to
# directly reuse the callback mechanism on weakref, we need the weakref
# and the key to be exactly the same object.  Reusing the callback mechanism
# minimizes the divergence between our implementation and Lib/weakref.py
#
# NB: Prefer using this when working with weakrefs of Tensors; e.g., do
# WeakIdRef(tensor) rather than weakref.ref(tensor); it handles a number of
# easy to get wrong cases transparently for you.
class WeakIdRef(weakref.ref):
    # [CLASS DOC] Weak reference whose hash and equality follow object identity instead of the
    # referent's own methods. Caches the referent's id at creation as the hash. Two refs are
    # equal when both referents are alive and the same object, else only if the same ref.
    __slots__ = ["_id"]

    def __init__(self, key, callback=None) -> None:
        # [FUNCTION DOC] Create a weak reference to key with an optional death callback and
        # cache the key's identity as the hash value.
        # Unlike stock weakref, which preserves hash semantics of the
        # original object but lazily defers hash calls until the first
        # time the user attempts to hash the weakref, we can eagerly
        # cache the id of the key as we know this is definitely the hash
        # method
        self._id = id(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self):
        # [FUNCTION DOC] Return the referent or None if dead; for tensors, first repairs the
        # Python object link (resurrection handling) by calling its weakref-fix method.
        r = super().__call__()
        # Special logic for Tensor PyObject resurrection
        if hasattr(r, "_fix_weakref"):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        # [FUNCTION DOC] Return the cached identity of the referent.
        return self._id

    def __eq__(self, other):
        # [FUNCTION DOC] Equal if both referents are alive and identical; if either is dead,
        # equal only when other is this very ref. Other must be callable like a weak ref.
        # An attractive but wrong alternate implementation is to only test if
        # the stored _ids match.  This can lead to an ABA problem if you have:
        #
        #   a1 = A()
        #   w1 = WeakIdRef(a1)
        #   del a1
        #   a2 = A()  # suppose it gets the same ID as a1
        #   w2 = WeakIdRef(a2)
        #   print(w1 == w2)
        #
        # This should be False, as a1 and a2 are unrelated (and a1 is
        # dead anyway)
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other


# This is the same as WeakIdRef but equality is checked using hash() rather than id.
# This will be equivalent to the one above except for classes where hash is not their id.
class _WeakHashRef(weakref.ref):
    # [CLASS DOC] HELPER: Like WeakIdRef but caches and compares the referent's hash value
    # instead of its identity (for objects whose hash stands in for identity, e.g. script
    # objects). Same dead-referent behavior: equal only to itself.
    __slots__ = ["_id"]

    def __init__(self, key, callback=None) -> None:
        # [FUNCTION DOC] HELPER: Create a weak reference to key and cache hash(key) in the
        # id slot (despite the name, it holds a hash value, not an identity).
        # Unlike stock weakref, which preserves hash semantics of the
        # original object but lazily defers hash calls until the first
        # time the user attempts to hash the weakref, we can eagerly
        # cache the id of the key as we know this is definitely the hash
        # method
        self._id = hash(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self):
        # [FUNCTION DOC] HELPER: Return the referent (or None), repairing tensor weakref
        # state first when the referent supports it.
        r = super().__call__()
        # Special logic for Tensor PyObject resurrection
        if hasattr(r, "_fix_weakref"):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Return the cached hash of the referent.
        return self._id

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal if both referents are alive and have equal hashes;
        # otherwise equal only when other is this very ref.
        # Use hash equality to determine ref equality.
        # ScriptObject implements __hash__ to return the wrapped IValue's id, so
        # this is equivalent to doing an identity comparison.
        a = self()
        b = other()
        if a is not None and b is not None:
            return hash(a) == hash(b)
        return self is other


# This is directly adapted from cpython/Lib/weakref.py
class WeakIdKeyDictionary(MutableMapping):
    # [CLASS DOC] Mutable mapping with weakly held keys compared by identity (or by hash when
    # ref_type is changed). Entries vanish when a key is garbage collected. State: data
    # (mapping from ref objects to values), ref_type, a set of active iteration guards, and a
    # list of removals deferred while iterating. Adapted from the standard library
    # WeakKeyDictionary; lines marked CHANGED differ.
    def __init__(self, dict=None, ref_type=WeakIdRef) -> None:  # CHANGED
        # [FUNCTION DOC] Create an empty mapping using ref_type to wrap keys, installing a
        # removal callback that defers deletions during iteration. If dict is given, its
        # entries are inserted. Keys must be weakly referenceable.
        self.data = {}

        self.ref_type = ref_type  # CHANGED

        def remove(k, selfref=ref(self)) -> None:
            # [FUNCTION DOC] HELPER: Death callback for a key ref. Queues the dead ref for
            # later removal if an iteration is active, else deletes its entry (ignoring
            # already-missing entries). Does nothing if the dictionary itself is gone.
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass

        self._remove = remove
        # A list of dead weakrefs (keys to be removed)
        self._pending_removals = []
        self._iterating = set()
        self._dirty_len = False
        if dict is not None:
            self.update(dict)

    def _commit_removals(self) -> None:
        # [FUNCTION DOC] HELPER: Drain the pending-removal list, deleting each entry and
        # ignoring entries that were already removed.
        # NOTE: We don't need to call this method before mutating the dict,
        # because a dead weakref never compares equal to a live weakref,
        # even if they happened to refer to equal objects.
        # However, it means keys may already have been removed.
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return

            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self) -> None:
        # [FUNCTION DOC] HELPER: Drop pending removals whose entries were already deleted
        # explicitly, and clear the dirty-length flag.
        d = self.data
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False

    def __delitem__(self, key) -> None:
        # [FUNCTION DOC] Delete the entry for key (by identity); raises KeyError if absent.
        # Marks the length as possibly stale.
        self._dirty_len = True
        del self.data[self.ref_type(key)]  # CHANGED

    def __getitem__(self, key):
        # [FUNCTION DOC] Return the value for key (by identity); raises KeyError if absent.
        return self.data[self.ref_type(key)]  # CHANGED

    def __len__(self) -> int:
        # [FUNCTION DOC] Number of live entries: stored entries minus those pending removal,
        # after scrubbing stale pending removals if an explicit delete happened.
        if self._dirty_len and self._pending_removals:
            # self._pending_removals may still contain keys which were
            # explicitly removed, we have to scrub them (see issue #21173).
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Short representation with class name and address; no contents.
        return f"<{self.__class__.__name__} at {id(self):#x}>"

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC] Store value under a weak, identity-compared key; the entry is
        # removed automatically when the key dies.
        self.data[self.ref_type(key, self._remove)] = value  # CHANGED

    def copy(self):
        # [FUNCTION DOC] Shallow copy containing only entries whose keys are alive; values
        # are shared. Returns a plain WeakIdKeyDictionary (the ref_type is not carried over).
        new = WeakIdKeyDictionary()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = value
        return new

    __copy__ = copy

    def __deepcopy__(self, memo):
        # [FUNCTION DOC] Copy with live keys shared and values deep-copied using the memo
        # mapping. Result has the same class as this object.
        from copy import deepcopy

        new = self.__class__()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = deepcopy(value, memo)
        return new

    def get(self, key, default=None):
        # [FUNCTION DOC] Return the value for key by identity, or default if absent.
        return self.data.get(self.ref_type(key), default)  # CHANGED

    def __contains__(self, key) -> bool:
        # [FUNCTION DOC] True if key is present by identity; False if key cannot be weakly
        # referenced.
        try:
            wr = self.ref_type(key)  # CHANGED
        except TypeError:
            return False
        return wr in self.data

    def items(self):
        # [FUNCTION DOC] Generator of (key, value) pairs for live keys; removals are deferred
        # while the generator is active.
        with _IterationGuard(self):
            for wr, value in self.data.items():
                key = wr()
                if key is not None:
                    yield key, value

    def keys(self):
        # [FUNCTION DOC] Generator of live keys; removals are deferred while it is active.
        # Also used as the iterator.
        with _IterationGuard(self):
            for wr in self.data:
                obj = wr()
                if obj is not None:
                    yield obj

    __iter__ = keys

    def values(self):
        # [FUNCTION DOC] Generator of values whose keys are still alive; removals deferred
        # while it is active.
        with _IterationGuard(self):
            for wr, value in self.data.items():
                if wr() is not None:
                    yield value

    def keyrefs(self):
        # [FUNCTION DOC] Return a list of the stored key refs (some may be dead).
        """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
        return list(self.data)

    def popitem(self):
        # [FUNCTION DOC] Remove and return an arbitrary (key, value) pair with a live key,
        # discarding dead entries on the way. Raises KeyError when empty.
        self._dirty_len = True
        while True:
            key, value = self.data.popitem()
            o = key()
            if o is not None:
                return o, value

    # pyrefly: ignore [bad-override]
    def pop(self, key, *args):
        # [FUNCTION DOC] Remove key and return its value; an optional extra argument is the
        # default returned if absent, otherwise KeyError is raised.
        self._dirty_len = True

        return self.data.pop(self.ref_type(key), *args)  # CHANGED

    def setdefault(self, key, default=None):
        # [FUNCTION DOC] Return the value for key, inserting default first if absent.
        return self.data.setdefault(
            self.ref_type(key, self._remove), default
        )  # CHANGED

    def update(self, dict=None, **kwargs) -> None:  # type: ignore[override]
        # [FUNCTION DOC] Insert entries from a mapping or an iterable of pairs, then from
        # keyword arguments (whose string keys are wrapped as weak keys, so strings fail
        # with TypeError since they cannot be weakly referenced).
        d = self.data
        if dict is not None:
            if not hasattr(dict, "items"):
                dict = type({})(dict)
            for key, value in dict.items():
                d[self.ref_type(key, self._remove)] = value  # CHANGED
        if kwargs:
            self.update(kwargs)

    def __ior__(self, other):
        # [FUNCTION DOC] In-place merge: update from other and return self.
        self.update(other)
        return self

    def __or__(self, other):
        # [FUNCTION DOC] Return a copy updated with entries of other if other is a mapping,
        # else NotImplemented.
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        # [FUNCTION DOC] Reflected merge: new dictionary of this class holding other's
        # entries then this one's (this one wins). NotImplemented if other is not a mapping.
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

    # Default Mapping equality will tests keys for equality, but
    # we want to test ids for equality
    def __eq__(self, other):
        # [FUNCTION DOC] Compare with another mapping by the identity of keys and equality of
        # values; NotImplemented for non-mappings.
        # KNOWN ISSUE: defining equality without a hash method makes instances unhashable.
        if not isinstance(other, Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {
            id(k): v for k, v in other.items()
        }


# Convenience alias
WeakTensorKeyDictionary = WeakIdKeyDictionary


class TensorWeakRef:
    # [CLASS DOC] Weak reference holder for a single tensor. Stores a standard weak
    # reference in the ref attribute; calling the instance returns the live tensor (after
    # the required weakref fix-up) or None if the tensor died.
    """Wrapper around a weak ref of a Tensor that handles the _fix_weakref() call required when unwrapping a Tensor weakref."""

    ref: WeakRef[Tensor]

    def __init__(self, tensor: Tensor) -> None:
        # [FUNCTION DOC] Create the weak reference; raises AssertionError if the argument
        # is not a tensor.
        if not isinstance(tensor, Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(tensor)}.")
        self.ref = weakref.ref(tensor)

    def __call__(self):
        # [FUNCTION DOC] Return the referenced tensor, or None if it was freed. Calls the
        # tensor's weakref fix-up before returning. Raises AssertionError if the referent
        # is not a tensor.
        out = self.ref()
        if out is None:
            return out
        if not isinstance(out, Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(out)}.")
        # TODO, add _fix_weakref type binding
        out._fix_weakref()  # type: ignore[attr-defined]
        return out
