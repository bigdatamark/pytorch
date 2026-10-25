# [MODULE DOC]
# Tracks object mutation (to invalidate cached guarded code) and detects
# dynamically created nn.Module instances via generation tagging. Contains
# MutationTracker, GenerationTracker, watch, ensure_patched, is_dynamic_nn_module
# and install_generation_tagging_init.
# Import-time side effect: records the original torch.nn.Module.__init__ in
# unpatched_nn_module_init. Patching only happens when functions are called.
"""Mutation tracking and dynamic module detection system for Dynamo.

This module provides mechanisms to track and respond to mutations in PyTorch modules
and detect dynamically created or modified modules.

Key components:
- MutationTracker: Tracks mutations to objects and invalidates associated cached code
- GenerationTracker: Tracks module creation timing to identify dynamic instances
- Patching system for nn.Module to detect mutations and dynamic creation

The system ensures that Dynamo's optimizations remain valid by detecting and responding
to runtime changes in module state and structure.
"""

import functools
import inspect
import weakref
from collections.abc import MutableMapping
from typing import Any

import torch.nn
from torch.nn import Module

from . import config
from .utils import ExactWeakKeyDictionary, nn_module_has_global_hooks


unpatched_nn_module_init = torch.nn.Module.__init__


class MutationTracker:
    # [CLASS DOC]
    # Per-object mutation watcher. Class attribute db is a shared weak-keyed
    # mapping from watched object to its tracker. Instance state: mutation_count
    # and watchers (weak references to guarded code). On mutation, all watchers
    # are invalidated once and the list is cleared.
    db: ExactWeakKeyDictionary = ExactWeakKeyDictionary()

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes mutation_count to zero and an empty watcher list.
        self.mutation_count: int = 0
        self.watchers: list[weakref.ReferenceType[Any]] = []

    def on_mutation(self, name: str) -> None:
        # [FUNCTION DOC]
        # Called when attribute name is set on the tracked object. Bumps the
        # mutation count, empties the watcher list, and calls invalidate (passing
        # the weak reference) on each watcher still alive. The name is unused.
        self.mutation_count += 1
        tmp = self.watchers
        self.watchers = []
        for ref in tmp:
            guarded = ref()
            if guarded is not None:
                guarded.invalidate(ref)

    def track(self, guarded_code: Any) -> None:
        # [FUNCTION DOC]
        # Registers a weak reference to guarded_code as a watcher.
        self.watchers.append(weakref.ref(guarded_code))


def watch(obj: Any, guarded_code: Any) -> None:
    # [FUNCTION DOC]
    # Arranges for guarded_code to be invalidated when obj is next mutated.
    # Patches the setattr of obj's type, creates a tracker for obj in the shared
    # db if missing, and registers guarded_code. Mutates global class state.
    """invalidate guarded_code when obj is mutated"""
    ensure_patched(type(obj))

    if obj not in MutationTracker.db:
        MutationTracker.db[obj] = MutationTracker()
    tracker = MutationTracker.db[obj]
    tracker.track(guarded_code)


def ensure_patched(cls: Any) -> None:
    # [FUNCTION DOC]
    # Idempotently replaces the class's setattr with a wrapper that notifies the
    # tracker of the instance (if any) before performing the original set. Uses a
    # marker attribute on the class to avoid double patching. Mutates cls.
    # KNOWN ISSUE: the marker is looked up with getattr, so a subclass of an
    # already-patched class inherits the marker and is not patched itself,
    # relying on inheriting the patched setattr.
    if getattr(cls, "___needs_mutation_patch", True):
        cls.___needs_mutation_patch = False
        original_setattr = cls.__setattr__

        @functools.wraps(original_setattr)
        def custom_setattr(self: Any, key: str, value: Any) -> None:
            # [FUNCTION DOC] HELPER: Notifies the tracker registered for self (if
            # none, ignores) that key is mutated, then delegates to the original
            # setattr and returns its result.
            try:
                MutationTracker.db[self].on_mutation(key)
            except KeyError:
                pass
            return original_setattr(self, key, value)

        cls.__setattr__ = custom_setattr


class GenerationTracker:
    # [CLASS DOC]
    # Static registry (class-level state only) used to identify nn.Modules
    # created dynamically. generation is a global counter; generation_values maps
    # objects to the generation at which they were created; dynamic_classes marks
    # module classes forced to be treated as dynamic.
    generation: int = 0
    dynamic_classes: ExactWeakKeyDictionary = ExactWeakKeyDictionary()
    generation_values: ExactWeakKeyDictionary = ExactWeakKeyDictionary()

    @classmethod
    def tag(cls, obj: Any) -> None:
        # [FUNCTION DOC]
        # Records the current generation for obj in the weak-keyed mapping.
        cls.generation_values[obj] = cls.generation

    @staticmethod
    def mark_class_dynamic(cls: type[torch.nn.Module]) -> None:
        # [FUNCTION DOC]
        # Marks a torch.nn.Module subclass as always dynamic. Raises
        # AssertionError if cls is not a Module subclass.
        if not issubclass(cls, torch.nn.Module):
            raise AssertionError(f"Expected a torch.nn.Module subclass, got {cls}")
        GenerationTracker.dynamic_classes[cls] = True

    @classmethod
    def get_generation_value(cls, obj: Any) -> int:
        # [FUNCTION DOC]
        # Returns the generation recorded for obj, or -1 if it was never tagged.
        if obj not in cls.generation_values:
            return -1
        return cls.generation_values[obj]

    @classmethod
    def check(cls, obj: Any) -> bool:
        # [FUNCTION DOC]
        # Returns true if obj was tagged during the current generation, meaning it
        # was created since the last generation bump.
        return (
            obj in cls.generation_values
            and cls.generation_values[obj] == cls.generation
        )

    @classmethod
    def clear(cls) -> None:
        # [FUNCTION DOC]
        # Resets the generation to zero and replaces both weak mappings with
        # empty ones. Mutates class-level state.
        cls.generation = 0
        cls.dynamic_classes = ExactWeakKeyDictionary()
        cls.generation_values = ExactWeakKeyDictionary()


def is_dynamic_nn_module(obj: Any, is_export: bool) -> bool:
    # [FUNCTION DOC]
    # Decides whether obj should be treated as a dynamic (not inlined/static)
    # module. True if: a Module with an instance-level forward or that is also a
    # mapping; the object has a truthy static torchdynamo_force_dynamic attribute
    # (its value is returned); a Module when not exporting or when free tensors
    # are installed; a Module while global module hooks exist; its class was
    # marked dynamic; or it was created in the current generation. Read-only.
    # KNOWN ISSUE: the force-dynamic attribute value is returned as is, so a
    # non-bool may leak out; and the is-mapping check is on a Module, which is
    # rare.
    """Check for nn.Modules() created dynamically or mutated"""
    if isinstance(obj, torch.nn.Module) and (
        "forward" in obj.__dict__ or isinstance(obj, (dict, MutableMapping))
    ):
        # A monkey patched `.forward` indicates something wacky is going on
        # Similarly a nn module also subclassed as a dict is unusual.
        return True
    try:
        return inspect.getattr_static(obj, "torchdynamo_force_dynamic")
    except AttributeError:
        pass
    if isinstance(obj, torch.nn.Module) and (
        not is_export or config.install_free_tensors
    ):
        return True

    if isinstance(obj, torch.nn.Module) and nn_module_has_global_hooks():
        return True
    dyn = GenerationTracker.dynamic_classes.get(type(obj)) or GenerationTracker.check(
        obj
    )
    return dyn


def install_generation_tagging_init() -> None:
    # [FUNCTION DOC]
    # One-time patches Module init and setstate so every new or unpickled module
    # is tagged with the current generation, then (every call) bumps the global
    # generation. Marks the patch as done with a flag on Module. Global side
    # effects on torch.nn.Module.
    """
    Monkey patch torch.nn.Module.__init__ and torch.nn.Module.__setstate__
    so we can detect nn.Module instances created dynamically inside forward methods.
    """

    if getattr(Module, "___needs_generation_tag_patch", True):
        init = Module.__init__

        def patched_init(self: Module, *args: Any, **kwargs: Any) -> None:
            # [FUNCTION DOC] HELPER: Runs the original Module init with all
            # arguments, then tags self with the current generation.
            init(self, *args, **kwargs)
            GenerationTracker.tag(self)

        Module.__init__ = patched_init  # type: ignore[method-assign]

        setstate = Module.__setstate__

        def patched_setstate(self: Module, state: Any) -> None:
            # [FUNCTION DOC] HELPER: Runs the original setstate with state, then
            # tags self with the current generation.
            setstate(self, state)
            GenerationTracker.tag(self)

        Module.__setstate__ = patched_setstate  # type: ignore[method-assign]

        Module.___needs_generation_tag_patch = False  # type: ignore[attr-defined]

    GenerationTracker.generation += 1
