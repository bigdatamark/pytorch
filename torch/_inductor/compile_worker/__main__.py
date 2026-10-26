# mypy: allow-untyped-defs
# [MODULE DOC] Command line entry point for the Inductor compile worker sidecar process launched by SubprocPool.
# Importing this module sets the triton ptxas path and preloads triton if available; running it as a script calls main().
import argparse
import base64
import functools
import importlib
import logging
import os
import sys
from typing import TypeVar

from torch._inductor.async_compile import pre_fork_setup
from torch._inductor.codecache import torch_key
from torch._inductor.compile_worker.subproc_pool import (
    SubprocKind,
    SubprocMain,
    SubprocPickler,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path


_T = TypeVar("_T")


log = logging.getLogger(__name__)

_set_triton_ptxas_path()

try:
    import triton

    if triton is None:
        raise AssertionError("triton failed to preload in parent")
except ImportError:
    pass


def _lookup_and_create_type(base: type[_T], qname: str) -> _T:
    # [FUNCTION DOC] HELPER: Resolve a dotted qualified name to a class by importing its module and
    # reading the attribute, verify it is a subclass of base, and return a new instance built with no arguments.
    # Raises TypeError if the class is not a subtype of base; import/attribute errors propagate.
    """
    Given a base type and qualified name: import & lookup that name, check
    that it's of the given type and then instantiate it.
    """
    pkg, name = qname.rsplit(".", 1)
    mod = importlib.import_module(pkg)
    ty = getattr(mod, name)
    if not issubclass(ty, base):
        raise TypeError(f"Type {ty} is not a subtype of {base}")
    return ty()


def main():
    # [FUNCTION DOC] Entry point of the compile worker sidecar subprocess. Parses command line options
    # (pickler class name, subprocess kind, worker count, parent pid, read/write file descriptors, base64 torch key),
    # exits silently if the parent pid no longer matches (parent died), opens the pipes, runs pre-fork setup,
    # installs the torch cache key, initializes the worker environment and runs the SubprocMain loop.
    # Any exception is logged, not raised. Side effects: file descriptors opened, global torch_key set.
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pickler", type=functools.partial(_lookup_and_create_type, SubprocPickler)
        )
        parser.add_argument("--kind", type=SubprocKind)
        parser.add_argument("--workers", type=int)
        parser.add_argument("--parent", type=int)
        parser.add_argument("--read-fd", type=int)
        parser.add_argument("--write-fd", type=int)
        parser.add_argument("--torch-key", type=str)
        args = parser.parse_args()
        if os.getppid() != args.parent:
            sys.exit(0)
        read_fd = os.fdopen(args.read_fd, "rb")
        write_fd = os.fdopen(args.write_fd, "wb")

        pre_fork_setup()

        torch_key.set(base64.b64decode(args.torch_key.encode("utf-8")))  # type: ignore[attr-defined]

        _async_compile_initializer(args.parent)

        SubprocMain(args.pickler, args.kind, args.workers, read_fd, write_fd).main()
    except Exception:
        log.exception("Uncaught exception in compile_worker subprocess")


if __name__ == "__main__":
    main()
