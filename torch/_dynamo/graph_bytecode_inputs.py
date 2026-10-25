# [MODULE DOC] Sidetable registry for objects that cannot be explicit FX graph inputs. Objects are
# registered under a dynamo-generated integer index; bytecode constructors (to re-create them in
# pre-graph bytecode) and weak references (to fetch them at graph run time) are kept in
# module-level dicts, plus a keep_alive list. Import-time side effect: registers
# get_external_object_by_index as the leaf-function module retriever in the higher-order ops.
import weakref
from collections.abc import Callable
from typing import Any

from torch._dynamo.source import Source


PyCodegen = Any

# This file is to handle types that we don't want to support
# as explicit FX graph inputs. This uses a sidetable which
# we populate in bytecode and is loaded during graph execution

# We use a dynamo-generated index as a level of indirection
# this allows us to register objects externally in pre-graph bytecode that we want
# to pass to the graph, but not support their types as graph inputs
index_to_bytecode_constructor: dict[int, Callable[[PyCodegen], None]] = {}

index_to_external_object_weakref: dict[int, weakref.ReferenceType[Any]] = {}

keep_alive: list[Any] = []


def has_user_objects() -> bool:
    # [FUNCTION DOC] Returns True if any object has been registered (bytecode constructor table
    # is non-empty). No side effects.
    return bool(index_to_bytecode_constructor)


def stash_graph_created_object(obj: Any) -> Any:
    # [FUNCTION DOC] Appends obj to the global keep_alive list so it outlives graph execution,
    # and returns obj unchanged.
    keep_alive.append(obj)
    return obj


CURRENT_STREAM_INDEX = 0


def set_external_object_by_index(index: int, value: Any) -> None:
    # [FUNCTION DOC] Sets the registry entry for the given index to a weak reference to value, and
    # appends value to keep_alive (so the weakref stays valid). Mutates globals; returns nothing.
    """Update an entry in the external object registry at runtime."""
    keep_alive.append(value)
    index_to_external_object_weakref[index] = weakref.ref(value)


def get_external_object_by_index(index: int) -> Any:
    # [FUNCTION DOC] Dereferences the weak reference registered for index and returns the object.
    # Raises AssertionError if the index is unregistered or the object has been collected.
    # Pure read of global state.
    if index not in index_to_external_object_weakref:
        raise AssertionError("Index not registered in index_to_user_object_weakref")
    obj = index_to_external_object_weakref[index]()
    if obj is None:
        raise AssertionError("User object is no longer alive")
    return index_to_external_object_weakref[index]()


def store_user_object_weakrefs(*args: Any) -> None:
    # [FUNCTION DOC] Replaces the whole weakref registry with weak references to the positional
    # arguments, keyed by their position. Mutates the global registry; returns nothing. Raises
    # TypeError if an argument does not support weak references.
    global index_to_external_object_weakref
    index_to_external_object_weakref.clear()
    index_to_external_object_weakref.update(
        {i: weakref.ref(arg) for i, arg in enumerate(args)}
    )


def reset_user_object_tracking() -> None:
    # [FUNCTION DOC] Clears the bytecode-constructor table, weakref registry and keep_alive list.
    # Global-state reset; returns nothing.
    index_to_bytecode_constructor.clear()
    index_to_external_object_weakref.clear()
    keep_alive.clear()


def register_graph_created_object(
    example_value: Any, construct_fn: Callable[[int, PyCodegen], None]
) -> int:
    # [FUNCTION DOC] Registers an object created while tracing the graph. Keeps example_value
    # alive, assigns the next index (current table size), stores a constructor that calls
    # construct_fn with the index and a codegen, and stores a weakref to example_value.
    # Returns the index. If a weakref cannot be made, reports a graph break through the
    # unimplemented helper (raises).
    global index_to_bytecode_constructor
    global keep_alive
    keep_alive.append(example_value)
    index = len(index_to_bytecode_constructor)
    index_to_bytecode_constructor[index] = lambda cg: construct_fn(index, cg)
    try:
        index_to_external_object_weakref[index] = weakref.ref(example_value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to graph-created external object",
            context=f"user_object: {example_value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index


# Register a user object to be used in the graph
def register_user_object(value: Any, source: Source) -> int:
    # [FUNCTION DOC] Registers a user-provided object under the next index; its constructor
    # reconstructs it by code-generating the given source. Stores a weakref to value and returns
    # the index. If value cannot be weakly referenced, reports a graph break (raises).
    # KNOWN ISSUE: unlike register_graph_created_object, value is not added to keep_alive.
    global index_to_bytecode_constructor
    index = len(index_to_bytecode_constructor)
    index_to_bytecode_constructor[index] = lambda cg: cg(source)
    try:
        index_to_external_object_weakref[index] = weakref.ref(value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to User Object",
            context=f"user_object: {value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index


# Register a callback so invoke_leaf_function can retrieve nn.Module instances at runtime.
# We use a callback pattern instead of having invoke_leaf_function import get_external_object_by_index
# directly, because higher-order ops should not depend on dynamo (dynamo depends on them, not vice versa).
from torch._higher_order_ops.invoke_leaf_function import (
    set_leaf_function_module_retriever,
)


set_leaf_function_module_retriever(get_external_object_by_index)
