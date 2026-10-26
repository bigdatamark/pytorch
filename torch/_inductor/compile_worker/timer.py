# [MODULE DOC] Defines Timer, a small inactivity timer used by compile worker pools to run a callback
# after a period without events.
from collections.abc import Callable
from threading import Lock, Thread
from time import monotonic, sleep


class Timer:
    # [CLASS DOC] Inactivity timer. Records the time of the last event and, when no event has arrived for
    # duration seconds, calls a callback once from a background daemon thread. State: background_thread
    # (started lazily on first event), last_called, duration, sleep_time (half the duration), exit flag, lock.
    """
    This measures how long we have gone since last receiving an event and if it is greater than a set interval, calls a function.
    """

    def __init__(
        self,
        duration: int | float,  # Duration in seconds
        call: Callable[[], None],  # Function to call when we expire
    ) -> None:
        # [FUNCTION DOC] Store the duration and callback; no thread is started until the first record_call.
        # We don't start the background thread until we actually get an event.
        self.background_thread: Thread | None = None
        self.last_called: float | None = None
        self.duration = duration
        self.sleep_time = duration / 2
        self.call = call
        self.exit = False

        self.lock = Lock()

    def record_call(self) -> None:
        # [FUNCTION DOC] Note that an event just happened: lazily start the daemon checker thread if none is running
        # and set last_called to the current monotonic time. Thread safe via the lock.
        with self.lock:
            if self.background_thread is None:
                self.background_thread = Thread(
                    target=self.check, daemon=True, name="subproc_worker_timer"
                )
                self.background_thread.start()
            self.last_called = monotonic()

    def quit(self) -> None:
        # [FUNCTION DOC] Ask the checker thread to stop at its next wakeup by setting the exit flag.
        with self.lock:
            self.exit = True

    def check(self) -> None:
        # [FUNCTION DOC] Background thread loop. Sleeps for half the duration, then under the lock returns if exit was
        # requested, otherwise raises AssertionError if no event was ever recorded, and keeps looping while the last event
        # is still within the duration. On expiry it clears last_called and the thread handle, releases the lock,
        # calls the callback once and returns.
        while True:
            # We have to be sensitive on checking here, to avoid too much impact on cpu
            sleep(self.sleep_time)
            with self.lock:
                if self.exit:
                    return
                if self.last_called is None:
                    raise AssertionError("last_called must be set before timer check")
                if self.last_called + self.duration >= monotonic():
                    continue
                self.last_called = None
                self.background_thread = None

            # Releasing lock in case self.call() takes a very long time or is reentrant
            self.call()
            return
