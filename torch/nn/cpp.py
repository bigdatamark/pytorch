# mypy: allow-untyped-defs
# [MODULE DOC] Interop helpers between Python nn modules and the C++ frontend: a live view
# wrapper over a C++ ordered dictionary and an nn.Module subclass delegating to a C++ module.
"""Functionality for Python <-> C++ frontend inter-op."""

from torch import nn


class OrderedDictWrapper:
    # [CLASS DOC] Read-only live view of an attribute (an ordered dictionary) on a C++ module.
    # State: cpp_module and attr name. The dictionary is re-fetched on every access so C++-side
    # changes are visible; mapping methods (items, keys, values, iteration, length, membership,
    # item lookup) all delegate to it.
    """A wrapper around a C++ OrderedDict.

    It dynamically evaluates the OrderedDict getter on a bound C++ module, such
    that new changes on the C++ side are picked up. Otherwise accessing e.g.
    ``cpp_module._parameters`` just once would get a frozen copy of the parameters
    at the time of access. ``torch.nn.Module`` accesses ``_parameters`` et al. via ``self.__dict__``
    so using properties does not work.
    """

    def __init__(self, cpp_module, attr) -> None:
        # [FUNCTION DOC] Store the C++ module and the name of the dictionary attribute to wrap.
        self.cpp_module = cpp_module
        self.attr = attr

    @property
    def cpp_dict(self):
        # [FUNCTION DOC] Property returning the current dictionary fetched from the C++ module by
        # attribute name (fresh on each access).
        return getattr(self.cpp_module, self.attr)

    # Magic methods cannot be assigned dynamically and bypass ``getattr``, so we
    # must manually override them.

    def items(self):
        # [FUNCTION DOC] Return the items view of the live C++ dictionary.
        return self.cpp_dict.items()

    def keys(self):
        # [FUNCTION DOC] Return the keys of the live C++ dictionary.
        return self.cpp_dict.keys()

    def values(self):
        # [FUNCTION DOC] Return the values of the live C++ dictionary.
        return self.cpp_dict.values()

    def __iter__(self):
        # [FUNCTION DOC] Return an iterator over the live C++ dictionary.
        return self.cpp_dict.__iter__()

    def __len__(self) -> int:
        # [FUNCTION DOC] Return the number of entries in the live C++ dictionary.
        return self.cpp_dict.__len__()

    def __contains__(self, key) -> bool:
        # [FUNCTION DOC] Return whether key is present in the live C++ dictionary.
        return self.cpp_dict.__contains__(key)

    def __getitem__(self, key):
        # [FUNCTION DOC] Return the entry for key from the live C++ dictionary; lookup errors
        # propagate from the C++ binding.
        return self.cpp_dict.__getitem__(key)


class ModuleWrapper(nn.Module):
    # [CLASS DOC] nn.Module facade over a C++ frontend module. State: cpp_module; parameter,
    # buffer and module registries are replaced by live OrderedDictWrapper views, and every
    # public attribute of the C++ module is copied onto the wrapper at construction. The
    # training flag is delegated to the C++ module.
    """A subclass of ``torch.nn.Module`` that wraps a C++ frontend module and delegates all access."""

    def __init__(self, cpp_module) -> None:
        # [FUNCTION DOC] Wrap cpp_module: store it before the base constructor (so the training
        # setter works), call the base constructor, install live registry views, and copy each
        # public attribute of the C++ module onto self (snapshot at construction time).
        # Assign before the super class constructor so ``self.training`` can be
        # assigned to in the super class constructor.
        self.cpp_module = cpp_module
        super().__init__()
        self._parameters = OrderedDictWrapper(cpp_module, "_parameters")  # type: ignore[assignment]
        self._buffers: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_buffers")  # type: ignore[assignment]
        self._modules: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_modules")  # type: ignore[assignment]
        for attr in dir(cpp_module):
            # Skip magic methods and the three attributes above.
            if not attr.startswith("_"):
                setattr(self, attr, getattr(self.cpp_module, attr))

    def _apply(self, fn, recurse=True):
        # [FUNCTION DOC] Apply fn in place to the data of every parameter, its gradient (if any)
        # and every buffer, avoiding new graph nodes. Returns self. The recurse argument is
        # ignored (parameters and buffers are always enumerated recursively).
        for param in self.parameters():
            # Tensors stored in modules are graph leaves, and we don't
            # want to create copy nodes, so we have to unpack the data.
            param.data = fn(param.data)
            if param._grad is not None:
                param._grad.data = fn(param._grad.data)

        for buf in self.buffers():
            buf.data = fn(buf.data)

        return self

    # nn.Module defines training as a boolean
    @property  # type: ignore[override]
    # pyrefly: ignore [bad-override]
    def training(self):
        # [FUNCTION DOC] Property getter: return the training flag of the wrapped C++ module.
        return self.cpp_module.training

    @training.setter
    def training(self, mode) -> None:
        # [FUNCTION DOC] Property setter: switch the wrapped C++ module to the given training mode
        # by calling its train method (side effect on the C++ module).
        self.cpp_module.train(mode)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return the string representation of the wrapped C++ module.
        return self.cpp_module.__repr__()
