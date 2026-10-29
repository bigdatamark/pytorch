# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC]
# Utilities to patch a class so its methods and properties route through a torch-function
# handler. Defines the method/property type tuples used for detection; no import-time side
# effects beyond imports.

import functools
from collections.abc import Callable
from types import (
    BuiltinMethodType,
    FunctionType,
    GetSetDescriptorType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import Any


FUNC_TYPES = (
    FunctionType,
    MethodDescriptorType,
    BuiltinMethodType,
    WrapperDescriptorType,
)
PROPERTY_TYPES = (GetSetDescriptorType, property)


def _py_wrap_method(orig: Callable, __torch_function__: Callable) -> Callable:
    # [FUNCTION DOC] HELPER: Returns a function that forwards its arguments to the
    # torch-function handler as (orig, no types, args, kwargs), copying orig's name and doc.
    def impl(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Calls the torch-function handler with orig and the arguments.
        return __torch_function__(orig, None, args, kwargs)

    # Copy metadata using functools.update_wrapper for just __name__ and __doc__
    functools.update_wrapper(impl, orig, assigned=("__name__", "__doc__"), updated=())

    return impl


def wrap_type(to_patch: Any, pattern: type, __torch_function__: Callable) -> None:
    # [FUNCTION DOC] Mutates the class to_patch: collects attributes from pattern and its
    # bases (excluding object), skips special ones (dict, new, init, repr, weakref, doc,
    # module, dir) and ones to_patch already overrides, then replaces functions/descriptors
    # with wrappers and properties/getset descriptors with wrapped properties that route
    # through __torch_function__. Returns None.
    wrap_method = _py_wrap_method

    all: dict[str, Any] = {}
    for t in reversed(pattern.mro()[:-1]):  # skip object
        all.update(t.__dict__)

    def wrap_attr(orig: Any) -> property:
        # [FUNCTION DOC] HELPER: Builds a property whose getter routes the descriptor's
        # get method through the torch-function handler.
        return property(wrap_method(orig.__get__, __torch_function__))

    for name, obj in all.items():
        if name in (
            "__dict__",
            "__new__",
            "__init__",
            "__repr__",
            "__weakref__",
            "__doc__",
            "__module__",
            "__dir__",
        ):
            continue

        # skip things that have been overloaded
        # things that come from object like `__eq__` still need to be patched, however.
        if hasattr(to_patch, name) and getattr(to_patch, name) is not getattr(
            object, name, None
        ):
            continue

        if isinstance(obj, FUNC_TYPES):
            setattr(to_patch, name, wrap_method(obj, __torch_function__))
        elif isinstance(obj, PROPERTY_TYPES):
            setattr(to_patch, name, wrap_attr(obj))
