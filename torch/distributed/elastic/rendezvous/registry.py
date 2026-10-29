# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Registry wiring for rendezvous handlers: defines factories for the built-in backends (etcd, etcd-v2,
# c10d, static) and helpers that register them, plus entry-point discovery of out-of-tree handlers
# in the "torchrun.handlers" group. Public export: get_rendezvous_handler.
# No import-time side effects beyond creating a logger.

import logging
from importlib.metadata import entry_points

from .api import (
    rendezvous_handler_registry as handler_registry,
    RendezvousHandler,
    RendezvousParameters,
)
from .dynamic_rendezvous import create_handler


log = logging.getLogger(__name__)

__all__ = ["get_rendezvous_handler"]


def _create_static_handler(params: RendezvousParameters) -> RendezvousHandler:
    # [FUNCTION DOC] HELPER: Builds a static TCP rendezvous handler from params; imports the module lazily to avoid import cost.
    from . import static_tcp_rendezvous

    return static_tcp_rendezvous.create_rdzv_handler(params)


def _create_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    # [FUNCTION DOC] HELPER: Builds the legacy etcd rendezvous handler from params; lazily imports etcd_rendezvous.
    from . import etcd_rendezvous

    return etcd_rendezvous.create_rdzv_handler(params)


def _create_etcd_v2_handler(params: RendezvousParameters) -> RendezvousHandler:
    # [FUNCTION DOC] HELPER: Builds an etcd-v2 handler: creates the etcd backend and store from params, then wraps them in a dynamic rendezvous handler.
    from .etcd_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)


def _create_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    # [FUNCTION DOC] HELPER: Builds a c10d handler: creates the c10d (TCPStore) backend and store from params, then wraps them in a dynamic rendezvous handler.
    from .c10d_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)


def _register_default_handlers() -> None:
    # [FUNCTION DOC] HELPER: Registers the four built-in handler factories (etcd, etcd-v2, c10d, static) under their names in the global registry.
    # Mutates the registry; takes no args and returns nothing.
    handler_registry.register("etcd", _create_etcd_handler)
    handler_registry.register("etcd-v2", _create_etcd_v2_handler)
    handler_registry.register("c10d", _create_c10d_handler)
    handler_registry.register("static", _create_static_handler)


def _register_out_of_tree_handlers() -> None:
    # [FUNCTION DOC] HELPER: Looks up entry points in the "torchrun.handlers" group, loads each one, calls the loaded object to obtain a
    # handler factory and registers it under the entry point name. Any exception per plugin is logged as a warning (with traceback)
    # and skipped so one bad plugin does not block others. Mutates the registry.
    discovered_handler_generators = entry_points(group="torchrun.handlers")

    for handler_generator in discovered_handler_generators:
        try:
            # pyrefly: ignore [bad-index]
            get_handler = discovered_handler_generators[handler_generator.name].load()
            handler_registry.register(handler_generator.name, get_handler())
        except Exception:
            log.warning(
                "Exception while registering out of tree plugin %s: ",
                handler_generator.name,
                exc_info=True,
            )


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    # [FUNCTION DOC] Returns a RendezvousHandler for the given RendezvousParameters by delegating to the global registry, which picks the
    # factory matching the backend named in params.
    # Raises whatever the registry raises (e.g. ValueError for an unknown backend).
    # KNOWN ISSUE: the docstring example calls this with a backend name plus parameters, but the function takes only params.
    """
    Obtain a reference to a :py:class`RendezvousHandler`.

    Custom rendezvous handlers can be registered by

    ::

      from torch.distributed.elastic.rendezvous import rendezvous_handler_registry
      from torch.distributed.elastic.rendezvous.registry import get_rendezvous_handler


      def create_my_rdzv(params: RendezvousParameters):
          return MyCustomRdzv(params)


      rendezvous_handler_registry.register("my_rdzv_backend_name", create_my_rdzv)

      my_rdzv_handler = get_rendezvous_handler(
          "my_rdzv_backend_name", RendezvousParameters
      )
    """
    return handler_registry.create_handler(params)
