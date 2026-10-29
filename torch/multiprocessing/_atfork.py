# mypy: allow-untyped-defs
# [MODULE DOC] Cross-platform helper to run a callable in a child process after fork. Exports
# register_after_fork. Import-time: picks a platform-specific private registration function
# (the multiprocessing utility hook on Windows, the OS at-fork hook elsewhere).
import sys


__all__ = ["register_after_fork"]

if sys.platform == "win32":
    import multiprocessing.util as _util

    def _register(func):
        # [FUNCTION DOC] HELPER: Windows variant. Registers a wrapper with the multiprocessing
        # after-fork utility; the wrapper takes the registered object and calls func with no
        # arguments. Note the registered object is the registration function itself.
        def wrapper(arg):
            # [FUNCTION DOC] HELPER: Ignores its argument and calls func.
            func()

        _util.register_after_fork(_register, wrapper)

else:
    import os

    def _register(func):
        # [FUNCTION DOC] HELPER: Non-Windows variant. Registers func as an in-child after-fork hook.
        os.register_at_fork(after_in_child=func)


def register_after_fork(func):
    # [FUNCTION DOC] Registers a no-argument callable to run in the child process after a fork,
    # using the platform-specific registration. Side effect: adds a process-wide hook.
    # KNOWN ISSUE: the docstring note about python versions older than 3.7 is obsolete.
    """Register a callable to be executed in the child process after a fork.

    Note:
        In python < 3.7 this will only work with processes created using the
        ``multiprocessing`` module. In python >= 3.7 it also works with
        ``os.fork()``.

    Args:
        func (function): Function taking no arguments to be called in the child after fork

    """
    _register(func)
