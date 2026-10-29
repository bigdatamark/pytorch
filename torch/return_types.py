# [MODULE DOC] Exposes the structseq return types of torch operations (for example the
# named tuples returned by max or sort) taken from the C extension return types module.
# Import-time side effects: copies every non-dunder attribute into this module's globals,
# appends public names to the all-list and to all_return_types, and registers every
# tuple-subclass class with the pytree registry so they flatten and unflatten as sequences.
import inspect

import torch
from torch.utils._pytree import register_pytree_node, SequenceKey


__all__ = ["pytree_register_structseq", "all_return_types"]

all_return_types = []

# error: Module has no attribute "_return_types"
return_types = torch._C._return_types  # type: ignore[attr-defined]


def pytree_register_structseq(cls):
    # [FUNCTION DOC] Registers a structseq class with the pytree registry so instances
    # flatten into a list of their elements (no context) and unflatten by calling the class
    # on that list; keyed flattening uses sequence indices as keys. Mutates global pytree
    # state; returns None.
    def structseq_flatten(structseq):
        # [FUNCTION DOC] HELPER: Returns the elements as a list and no context.
        return list(structseq), None

    def structseq_flatten_with_keys(structseq):
        # [FUNCTION DOC] HELPER: Like flatten, but pairs each element with a sequence key.
        values, context = structseq_flatten(structseq)
        return [(SequenceKey(i), v) for i, v in enumerate(values)], context

    def structseq_unflatten(values, context):
        # [FUNCTION DOC] HELPER: Rebuilds the structseq class from the list of values.
        return cls(values)

    register_pytree_node(
        cls,
        structseq_flatten,
        structseq_unflatten,
        flatten_with_keys_fn=structseq_flatten_with_keys,
    )


for name in dir(return_types):
    if name.startswith("__"):
        continue

    _attr = getattr(return_types, name)
    globals()[name] = _attr

    if not name.startswith("_"):
        # pyrefly: ignore [unresolvable-dunder-all]
        __all__.append(name)
        all_return_types.append(_attr)

    # Today everything in torch.return_types is a structseq, aka a "namedtuple"-like
    # thing defined by the Python C-API. We're going to need to modify this when that
    # is no longer the case.
    # NB: I don't know how to check that something is a "structseq" so we do a fuzzy
    # check for tuple
    if inspect.isclass(_attr) and issubclass(_attr, tuple):
        pytree_register_structseq(_attr)
