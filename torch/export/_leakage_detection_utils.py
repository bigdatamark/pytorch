# [MODULE DOC]
# Debugging utilities that use the garbage collector's referrer graph to detect fake
# tensors that leaked out of non-strict export tracing. Contains small predicates and
# find_legit_leaks_from_referrers. No import-time side effects beyond imports and
# defining the _SKIP_TYPES constant.
import gc
import types
import typing
import weakref
from typing_extensions import TypeIs

from torch.fx.experimental.symbolic_shapes import TrackedFake


"""
These functions are used to detect potential fake tensor leakage when using PT2 export.
See NOTE [export non-strict fake tensor leak detection]

There are some complications that made this logic overly complicated:
1) Python 3.10 and Python 3.12 have different ways of implementing referrer so
   we need to account for whether it is ref.__dict__ or the real ref object

2) There are some internal PT2 references to fake tensors like `TrackedFake`
3) closures, generators, and bound methods can hold fake tensors.
4) global object can hold onto a fake tensor

In general, these utils are our last resort to detect fake tensors. if the leak happens
within the model attributes, we have a separate mechanism to detect. This tool relies a bit
on garbage collector internal details, so I think it is unsafe to turn on by default, hence
this tool should be used as debugging tool.
"""


# Things we never want to flag as leaks
_SKIP_TYPES = (
    types.FrameType,
    types.ModuleType,
)


def _is_globals_or_locals(obj: typing.Any) -> bool:
    # [FUNCTION DOC] HELPER: True if obj is this module's globals mapping or the current locals mapping.
    # These comparisons only make sense within this frame; still cheap to check.
    return obj is globals() or obj is locals()


def _is_tracked_fake(obj: typing.Any) -> TypeIs[TrackedFake]:
    # [FUNCTION DOC] HELPER: type guard, True if obj is a TrackedFake instance.
    return isinstance(obj, TrackedFake)


def _is_gm_meta_like_dict(d: dict, o: typing.Any) -> bool:
    # [FUNCTION DOC] HELPER: True if the dict d has an entry under the key "val" that is the very
    # object o (identity), i.e. it looks like an fx node meta mapping holding the fake.
    # Hope gm.meta was a custom dict we can assert on
    return d.get("val") is o


def _dict_is_attr_of_tracked_fake(d: dict) -> bool:
    # [FUNCTION DOC] HELPER: queries the garbage collector for referrers of d and returns True if
    # any is a TrackedFake whose instance attribute dict is exactly d; otherwise False.
    """
    Python 3.10 quirk: sometimes the referrer is obj.__dict__ instead of obj.
    Check if this dict is exactly the __dict__ of a TrackedFake.
    """
    for parent in gc.get_referrers(d):
        if (
            hasattr(parent, "__dict__")
            and parent.__dict__ is d
            and _is_tracked_fake(parent)
        ):
            return True
    return False


def find_legit_leaks_from_referrers(active_fakes: weakref.WeakSet) -> weakref.WeakSet:
    # [FUNCTION DOC]
    # Takes a weak set of live fake tensors and returns a new weak set containing those that are
    # still referenced by something that is not ignorable. Ignored referrers: the temporary list
    # made here, globals/locals mappings, frames, modules, TrackedFake objects, node-meta-like
    # dicts holding the fake under "val", and attribute dicts of TrackedFake. Any other referrer
    # (including other dicts) marks the fake as leaked. Relies on gc internals; debug use only.
    legit_leak: weakref.WeakSet = weakref.WeakSet()

    # This is so that we don't falsely flag generator to be holding fake tensor
    fake_list = list(active_fakes)
    fake_list_id = id(fake_list)

    for act in fake_list:
        # Track by id to avoid processing duplicate referrers
        seen = set()
        # Assume it's a leak unless we find only ignorable referrers
        flagged = False

        for r in gc.get_referrers(act):
            rid = id(r)
            if rid in seen:
                continue
            seen.add(rid)

            # Skip our own fake_list
            if rid == fake_list_id:
                continue

            # Fast-path: skip obvious non-owners
            if _is_globals_or_locals(r):
                continue
            if isinstance(r, _SKIP_TYPES):
                continue
            if _is_tracked_fake(r):
                # TrackedFake should be ignored
                continue

            # Handle dicts carefully (Python 3.10 sometimes shows __dict__)
            if isinstance(r, dict):
                if _is_gm_meta_like_dict(r, act):
                    continue
                if _dict_is_attr_of_tracked_fake(r):
                    continue
                flagged = True
                break

            # Any other referrer we don't explicitly whitelist counts as a leak
            flagged = True
            break

        if flagged:
            legit_leak.add(act)

    return legit_leak
