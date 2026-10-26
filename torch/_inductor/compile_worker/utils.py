# [MODULE DOC] Helpers for compile worker subprocesses: top-level process flag, worker initializer with parent-death
# watchdog, and parent change detection.
import os
import signal
from threading import Thread
from time import sleep

from torch._inductor.compile_worker import watchdog


_IN_TOPLEVEL_PROCESS = True


def in_toplevel_process() -> bool:
    # [FUNCTION DOC] Return True in the main process and False inside async compile worker subprocesses
    # (the flag is cleared by _async_compile_initializer).
    global _IN_TOPLEVEL_PROCESS
    return _IN_TOPLEVEL_PROCESS


# If this process dies abnormally (e.g. segfault)
# it will not shut down the workers. Instead,
# the workers will have their parent reassigned to the
# init process. This launches a separate thread to
# watch for the worker getting reassigned,
# and cleans it up in this case.
#
# This function cannot be an inner function since otherwise mp_context="spawn" would
# not work for ProcessPoolExecutor since inner functions cannot be pickled.
def _async_compile_initializer(orig_ppid: int, close_fds: tuple[int, ...] = ()) -> None:
    # [FUNCTION DOC] HELPER: Pool worker initializer; must be module level so spawn can pickle it.
    # Closes any inherited file descriptors in close_fds (fork mode only, ignoring OSError), records the original
    # parent pid, starts a daemon thread that checks once a minute and SIGKILLs this process if the parent changed,
    # ignores SIGINT, installs the torch crash handler, marks the process as not top-level, and claims a
    # watchdog heartbeat slot. Mutates module globals.
    # In fork mode a worker inherits the sidecar's entire fd table, including the
    # sidecar<->parent pipes. The worker must not keep those open: holding the
    # result pipe's write end would stop the parent from ever seeing EOF when the
    # sidecar dies. The caller only passes these fds under fork; under spawn the
    # worker doesn't inherit them and the same integers would name unrelated fds
    # the fresh interpreter reused, so closing them there would be silent
    # corruption (the OSError guard wouldn't catch it) rather than a no-op.
    for fd in close_fds:
        try:
            os.close(fd)
        except OSError:
            pass

    import torch._C

    def run() -> None:
        # [FUNCTION DOC] HELPER: Watchdog thread body; loops forever sleeping 60 seconds and kills this process with SIGKILL
        # if its parent pid differs from the original parent.
        while True:
            sleep(60)
            if orig_ppid != os.getppid():
                os.kill(os.getpid(), signal.SIGKILL)

    global _watchdog_thread, _original_parent
    _original_parent = orig_ppid
    _watchdog_thread = Thread(target=run, daemon=True)
    _watchdog_thread.start()
    # Ignore Ctrl-C (i.e. SIGINT) sent to pool workers to avoid meaningless log spam.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Install a crash handler to print out the stacktrace for SEGV
    torch._C._initCrashHandler()

    # Set a bit to distinguish async_compile subprocesses from the toplevel process.
    global _IN_TOPLEVEL_PROCESS
    _IN_TOPLEVEL_PROCESS = False

    # Claim a heartbeat slot so the sidecar watchdog can report this worker's
    # compile phase (no-op unless the sidecar allocated the shared buffer).
    watchdog.init_worker_slot()


_watchdog_thread: Thread | None = None
_original_parent: int | None = None


def has_parent_changed() -> bool:
    # [FUNCTION DOC] Return True if the current parent pid differs from the one recorded at initialization
    # (also True if the initializer never ran, since the recorded parent is then unset).
    return _original_parent != os.getppid()
