# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for step closures of the lazy backend (callbacks registered to run around a
# step marker), in synchronous and asynchronous mode, including exception propagation. Import-time
# side effect: initializes the TS lazy backend. Threading events are used as flags shared between
# the test body and the closures.

from threading import Event
from time import sleep

import torch._lazy
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, skipIfFreeThreaded, TestCase


torch._lazy.ts_backend.init()


class ClosuresTest(TestCase):
    # [CLASS DOC] Test case for step closures; each test uses a fresh event as a flag.
    def test_synchronous(self):
        # [FUNCTION DOC] Registers a synchronous closure that sleeps a second, checks the flag is
        # clear and sets it, calls the step marker, and asserts the flag is set right after, i.e.
        # the marker waited for the closure to finish.
        flag = Event()
        if flag.is_set():
            raise AssertionError("flag should not be set initially")

        def closure():
            # [FUNCTION DOC] HELPER: Sleeps one second, raises an assertion error if the flag is
            # already set, then sets the flag.
            sleep(1)
            if flag.is_set():
                raise AssertionError("flag should not be set during closure")
            flag.set()

        torch._lazy.add_step_closure(closure)
        torch._lazy.mark_step()

        # should not get to this part before closure is finished running
        if not flag.is_set():
            raise AssertionError("flag should be set after mark_step")

    def test_asynchronous(self):
        # [FUNCTION DOC] Registers an asynchronous closure that sleeps then asserts the flag was
        # set by the main thread, calls the step marker, asserts the flag is still clear (marker
        # did not wait), and then sets the flag for the closure to observe.
        # KNOWN ISSUE: the closure runs on a worker thread, so a failure of its assertion does not
        #     directly fail this test.
        flag = Event()
        if flag.is_set():
            raise AssertionError("flag should not be set initially")

        def closure():
            # [FUNCTION DOC] HELPER: Sleeps one second, then raises an assertion error if the flag
            # has not been set by the main thread.
            sleep(1)
            if not flag.is_set():
                raise AssertionError("flag should be set by the time closure runs")

        torch._lazy.add_step_closure(closure, run_async=True)
        torch._lazy.mark_step()

        # should get to this part and complete before closure is finished running
        if flag.is_set():
            raise AssertionError("flag should not be set yet (async)")
        flag.set()

    def test_synchronous_exception(self):
        # [FUNCTION DOC] Registers a synchronous closure that sets the flag and raises a runtime
        # error, then calls the step marker; the runtime error must propagate out of the marker
        # (reaching code after it raises an assertion error) and the flag must be set.
        flag = Event()
        if flag.is_set():
            raise AssertionError("flag should not be set initially")

        try:

            def closure():
                # [FUNCTION DOC] HELPER: Sets the flag and raises a runtime error.
                flag.set()
                raise RuntimeError("Simulating exception in closure")

            torch._lazy.add_step_closure(closure)
            torch._lazy.mark_step()

            raise AssertionError("Should not reach here")
        except RuntimeError:
            if not flag.is_set():
                raise AssertionError(
                    "Should have caught exception from closure"
                ) from None

    @skipIfFreeThreaded(
        "Non-deterministic, fails more consistently in free threaded python",
    )
    def test_asynchronous_exception(self):
        # [FUNCTION DOC] Registers an asynchronous closure that sets the flag and raises, steps,
        # and waits up to 5 seconds for the flag. Then registers a second asynchronous closure
        # and steps again, expecting a runtime error (from the earlier failed closure) to be
        # raised; finally asserts the flag is still set. Skipped under free-threaded Python
        # because it is non-deterministic.
        flag = Event()
        if flag.is_set():
            raise AssertionError("flag should not be set initially")

        def closure1():
            # [FUNCTION DOC] HELPER: Sets the flag and raises a runtime error; runs asynchronously.
            flag.set()
            raise RuntimeError("Simulating exception in closure1")

        torch._lazy.add_step_closure(closure1, run_async=True)
        torch._lazy.mark_step()

        flag.wait(timeout=5)

        try:

            def closure2():  # Should never execute
                # [FUNCTION DOC] HELPER: Clears the flag; expected never to execute because the
                # earlier closure failed.
                flag.clear()

            torch._lazy.add_step_closure(closure2, run_async=True)
            torch._lazy.mark_step()

            raise AssertionError("Should not reach here")
        except RuntimeError:
            # Should have caught exception from closure1
            pass

        if not flag.is_set():
            raise AssertionError("flag should still be set")


if __name__ == "__main__":
    run_tests()
