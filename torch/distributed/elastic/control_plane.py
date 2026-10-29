# [MODULE DOC] Control plane helper for elastic workers. Provides worker_main, a context manager
# decorator combining error recording with an optional C++ worker server listening on a unix
# socket named by the TORCH_WORKER_SERVER_SOCKET environment variable.
import os
from collections.abc import Generator
from contextlib import contextmanager, ExitStack

from torch.distributed.elastic.multiprocessing.errors import record


__all__ = [
    "worker_main",
]

TORCH_WORKER_SERVER_SOCKET = "TORCH_WORKER_SERVER_SOCKET"


@contextmanager
def _worker_server(socket_path: str) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that starts a _WorkerServer on the given unix socket
    # path, yields, and always shuts the server down on exit (including on error).
    from torch._C._distributed_c10d import _WorkerServer

    server = _WorkerServer(socket_path)
    try:
        yield
    finally:
        server.shutdown()


@record
@contextmanager
def worker_main() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager (also wrapped by the error record decorator so uncaught
    # exceptions are recorded). If the TORCH_WORKER_SERVER_SOCKET environment variable is set, a
    # worker server is run on that socket for the duration of the body; otherwise it just yields.
    # Intended to wrap a worker's main function. Exceptions propagate after cleanup.
    """
    This is a context manager that wraps your main entry function. This combines
    the existing ``errors.record`` logic as well as a new ``_WorkerServer`` that
    exposes handlers via a unix socket specified by
    ``TORCH_WORKER_SERVER_SOCKET``.

    Example

    ::

     @worker_main()
     def main():
         pass


     if __name__ == "__main__":
         main()

    """
    with ExitStack() as stack:
        socket_path = os.environ.get(TORCH_WORKER_SERVER_SOCKET)
        if socket_path is not None:
            stack.enter_context(_worker_server(socket_path))

        yield
