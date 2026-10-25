# [MODULE DOC]
# Defines ProxyValue, a generic holder tying a concrete value to its FX Proxy/Node.
# Used by ExportPassBase-style passes. No import-time side effects.
# pyre-strict
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

import torch


_T = TypeVar("_T")


class ProxyValue(Generic[_T]):
    # [CLASS DOC] Pairs a concrete (fake/example) value with the FX Proxy or Node
    # that represents it in the graph being built. State: data (the value, an
    # iterable or tensor) and proxy_or_node (either a Proxy or a Node).
    # Used by export passes to carry values and graph handles together.
    # pyre-ignore
    def __init__(self, data: Iterable[_T], proxy: torch.fx.Proxy | torch.fx.Node):
        # [FUNCTION DOC] Store the value and its graph handle (Proxy or Node) as given.
        # pyre-ignore
        self.data = data
        self.proxy_or_node = proxy

    @property
    def node(self) -> torch.fx.Node:
        # [FUNCTION DOC] Property returning the underlying Node: the stored Node itself,
        # or the Node of the stored Proxy. Raises AssertionError for any other type.
        if isinstance(self.proxy_or_node, torch.fx.Node):
            return self.proxy_or_node
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise AssertionError(
                f"expected Node or Proxy, got {type(self.proxy_or_node)}"
            )
        return self.proxy_or_node.node

    @property
    def proxy(self) -> torch.fx.Proxy:
        # [FUNCTION DOC] Property returning the stored Proxy. Raises RuntimeError if only
        # a Node is attached.
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise RuntimeError(
                f"ProxyValue doesn't have attached Proxy object. Node: {self.proxy_or_node.format_node()}"
            )
        return self.proxy_or_node

    def to_tensor(self) -> torch.Tensor:
        # [FUNCTION DOC] Return data as a Tensor; raises AssertionError if it is not one.
        if not isinstance(self.data, torch.Tensor):
            raise AssertionError(f"expected Tensor, got {type(self.data)}")
        return self.data

    def is_tensor(self) -> bool:
        # [FUNCTION DOC] Return whether the stored data is a Tensor.
        return isinstance(self.data, torch.Tensor)

    # pyre-ignore
    def __iter__(self) -> Iterator[_T]:
        # [FUNCTION DOC] Iterate over the elements of the stored data.
        yield from self.data

    def __bool__(self) -> bool:
        # [FUNCTION DOC] Truthiness of the stored data (not of the proxy).
        return bool(self.data)
