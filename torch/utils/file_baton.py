# mypy: allow-untyped-defs
# [MODULE DOC] Provides FileBaton, a minimal lock built on atomic creation of a lock file, used
# to coordinate processes (for example around shared build directories). No import-time effects.
import os
import time
import warnings


class FileBaton:
    # [CLASS DOC] File-based baton: whoever atomically creates the lock file holds it, and
    # removing the file releases it. State: lock_file_path, wait_seconds (poll interval),
    # fd (descriptor of the created file, None until acquired) and warn_after_seconds (optional
    # threshold for a one-time warning while waiting). Methods: try_acquire, wait, release.
    """A primitive, file-based synchronization utility."""

    def __init__(self, lock_file_path, wait_seconds=0.1, warn_after_seconds=None) -> None:
        # [FUNCTION DOC] Stores the lock file path, poll interval and optional warning threshold,
        # and initializes the descriptor to None. Performs no file I/O.
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
            warn_after_seconds: The seconds to wait before showing
                lock file path to warn existing lock file.
        """
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None
        self.warn_after_seconds = warn_after_seconds

    def try_acquire(self) -> bool | None:
        # [FUNCTION DOC] Attempts to create the lock file exclusively (create, fail if present).
        # On success stores the open descriptor and returns True; returns False if the file
        # already exists. Other OS errors propagate. Side effect: creates the file.
        """
        Try to atomically create a file under exclusive access.

        Returns:
            True if the file could be created, else False.
        """
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            return False

    def wait(self) -> None:
        # [FUNCTION DOC] Blocks, sleeping wait_seconds per iteration, until the lock file no
        # longer exists. If warn_after_seconds is set, emits one warning naming the lock file once
        # that much time has elapsed. Returns nothing; does not acquire the baton.
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
        has_warned = False

        start_time = time.time()
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

            if self.warn_after_seconds is not None:
                if time.time() - start_time > self.warn_after_seconds and not has_warned:
                    warnings.warn(f'Waited on lock file "{self.lock_file_path}" for '
                                  f'{self.warn_after_seconds} seconds.', stacklevel=2)
                    has_warned = True

    def release(self) -> None:
        # [FUNCTION DOC] Closes the stored descriptor if one exists, then deletes the lock file.
        # Raises FileNotFoundError if the file is already gone. The descriptor is not reset
        # to None afterwards.
        """Release the baton and removes its file."""
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)
