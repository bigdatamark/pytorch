# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]
# [MODULE DOC] Unit tests for torch.futures.Future: result and exception propagation, then and
# add_done_callback chaining and ordering, collect_all and wait_all helpers, pickling refusal.
# Uses threads to check cross-thread completion. No import-time side effects besides a TypeVar.

import threading
import time
import torch
import unittest
from torch.futures import Future
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase, TemporaryFileName, run_tests
from typing import TypeVar

T = TypeVar("T")


def add_one(fut):
    # [FUNCTION DOC] HELPER: Callback that blocks on the given future and returns its value plus
    # one. Used to build long chains of then callbacks on tensor futures.
    return fut.wait() + 1


class TestFuture(TestCase):
    # [CLASS DOC] Test case covering the Future API: set_result, set_exception, wait, value,
    # done, then, add_done_callback, collect_all, wait_all. Stateless; each test builds its own
    # futures.
    def test_set_exception(self) -> None:
        # [FUNCTION DOC] Checks that an exception set on a future is raised by wait and by
        # value, and that a then callback calling value on a failed future makes the chained
        # future raise a RuntimeError mentioning the underlying error.
        # This test is to ensure errors can propagate across futures.
        error_msg = "Intentional Value Error"
        value_error = ValueError(error_msg)

        f = Future[T]()  # type: ignore[valid-type]
        # Set exception
        # pyrefly: ignore [bad-argument-type]
        f.set_exception(value_error)
        # Exception should throw on wait
        with self.assertRaisesRegex(ValueError, "Intentional"):
            f.wait()

        # Exception should also throw on value
        f = Future[T]()  # type: ignore[valid-type]
        # pyrefly: ignore [bad-argument-type]
        f.set_exception(value_error)
        with self.assertRaisesRegex(ValueError, "Intentional"):
            f.value()

        def cb(fut):
            # [FUNCTION DOC] HELPER: Callback that reads the value of the future, which re-
            # raises its error.
            fut.value()

        f = Future[T]()  # type: ignore[valid-type]
        # pyrefly: ignore [bad-argument-type]
        f.set_exception(value_error)

        with self.assertRaisesRegex(RuntimeError, "Got the following error"):
            cb_fut = f.then(cb)
            cb_fut.wait()

    def test_set_exception_multithreading(self) -> None:
        # [FUNCTION DOC] Checks error propagation across threads: one thread waits while the
        # main thread sets an exception, and one thread chains a then callback and expects a
        # wrapped RuntimeError.
        # Ensure errors can propagate when one thread waits on future result
        # and the other sets it with an error.
        error_msg = "Intentional Value Error"
        value_error = ValueError(error_msg)

        def wait_future(f):
            # [FUNCTION DOC] HELPER: Thread body that asserts waiting on the future raises
            # ValueError.
            with self.assertRaisesRegex(ValueError, "Intentional"):
                f.wait()

        f = Future[T]()  # type: ignore[valid-type]
        t = threading.Thread(target=wait_future, args=(f, ))
        t.start()
        # pyrefly: ignore [bad-argument-type]
        f.set_exception(value_error)
        t.join()

        def cb(fut):
            # [FUNCTION DOC] HELPER: Callback that reads the value of the future, which re-
            # raises its error.
            fut.value()

        def then_future(f):
            # [FUNCTION DOC] HELPER: Thread body that chains the callback and asserts waiting on
            # the chained future raises a RuntimeError reporting the original error.
            fut = f.then(cb)
            with self.assertRaisesRegex(RuntimeError, "Got the following error"):
                fut.wait()

        f = Future[T]()  # type: ignore[valid-type]
        t = threading.Thread(target=then_future, args=(f, ))
        t.start()
        # pyrefly: ignore [bad-argument-type]
        f.set_exception(value_error)
        t.join()

    def test_done(self) -> None:
        # [FUNCTION DOC] Checks done is false for a fresh future and true after a result is set.
        f = Future[torch.Tensor]()
        self.assertFalse(f.done())

        f.set_result(torch.ones(2, 2))
        self.assertTrue(f.done())

    def test_done_exception(self) -> None:
        # [FUNCTION DOC] Checks that a then callback which raises still yields a completed
        # chained future, and that waiting on it raises the callback's RuntimeError.
        err_msg = "Intentional Value Error"

        def raise_exception(unused_future):
            # [FUNCTION DOC] HELPER: Callback that ignores its future and always raises a
            # RuntimeError.
            raise RuntimeError(err_msg)

        f1 = Future[torch.Tensor]()
        self.assertFalse(f1.done())
        f1.set_result(torch.ones(2, 2))
        self.assertTrue(f1.done())

        f2 = f1.then(raise_exception)
        self.assertTrue(f2.done())
        with self.assertRaisesRegex(RuntimeError, err_msg):
            f2.wait()

    def test_wait(self) -> None:
        # [FUNCTION DOC] Checks wait returns the tensor that was set as the result.
        f = Future[torch.Tensor]()
        f.set_result(torch.ones(2, 2))

        self.assertEqual(f.wait(), torch.ones(2, 2))

    def test_wait_multi_thread(self) -> None:
        # [FUNCTION DOC] Checks wait blocks until another thread sets the result after a short
        # sleep.

        def slow_set_future(fut, value):
            # [FUNCTION DOC] HELPER: Thread body that sleeps half a second then sets the given
            # value.
            time.sleep(0.5)
            fut.set_result(value)

        f = Future[torch.Tensor]()

        t = threading.Thread(target=slow_set_future, args=(f, torch.ones(2, 2)))
        t.start()

        self.assertEqual(f.wait(), torch.ones(2, 2))
        t.join()

    def test_mark_future_twice(self) -> None:
        # [FUNCTION DOC] Checks setting a result twice raises a RuntimeError saying a future can
        # only be marked completed once.
        fut = Future[int]()
        fut.set_result(1)
        with self.assertRaisesRegex(
            RuntimeError,
            "Future can only be marked completed once"
        ):
            fut.set_result(1)

    def test_pickle_future(self):
        # [FUNCTION DOC] Checks that saving a future with torch.save raises a RuntimeError
        # stating that futures cannot be pickled. Uses a temporary file name.
        fut = Future[int]()
        errMsg = "Can not pickle torch.futures.Future"
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, errMsg):
                torch.save(fut, fname)

    def test_then(self):
        # [FUNCTION DOC] Checks a then callback receives the future and its result is the
        # chained future's value (original value plus one).
        fut = Future[torch.Tensor]()
        then_fut = fut.then(lambda x: x.wait() + 1)

        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        self.assertEqual(then_fut.wait(), torch.ones(2, 2) + 1)

    def test_chained_then(self):
        # [FUNCTION DOC] Chains twenty then callbacks and checks the i-th chained future yields
        # the original tensor plus i+1 after the root result is set.
        fut = Future[torch.Tensor]()
        futs = []
        last_fut = fut
        for _ in range(20):
            last_fut = last_fut.then(add_one)
            futs.append(last_fut)

        fut.set_result(torch.ones(2, 2))

        for i in range(len(futs)):
            self.assertEqual(futs[i].wait(), torch.ones(2, 2) + i + 1)

    def _test_then_error(self, cb, errMsg):
        # [FUNCTION DOC] HELPER: Registers the given callback via then on an integer future,
        # completes it with 5, checks the original future is unaffected, and checks waiting on
        # the chained future raises a RuntimeError matching the given message pattern.
        fut = Future[int]()
        then_fut = fut.then(cb)

        fut.set_result(5)
        self.assertEqual(5, fut.wait())
        with self.assertRaisesRegex(RuntimeError, errMsg):
            then_fut.wait()

    def test_then_wrong_arg(self):
        # [FUNCTION DOC] Checks a callback that treats its argument as a tensor fails, because
        # it is given the future itself, producing an unsupported operand error.

        def wrong_arg(tensor):
            # [FUNCTION DOC] HELPER: Callback that adds one to its argument; fails when given a
            # future.
            return tensor + 1

        self._test_then_error(wrong_arg, "unsupported operand type.*Future.*int")

    def test_then_no_arg(self):
        # [FUNCTION DOC] Checks a callback taking no parameters fails with a positional argument
        # count error.

        def no_arg():
            # [FUNCTION DOC] HELPER: Callback with no parameters that returns True.
            return True

        self._test_then_error(no_arg, "takes 0 positional arguments but 1 was given")

    def test_then_raise(self):
        # [FUNCTION DOC] Checks a then callback raising ValueError surfaces as an error on the
        # chained future.

        def raise_value_error(fut):
            # [FUNCTION DOC] HELPER: Callback that raises a ValueError (Expected error).
            raise ValueError("Expected error")

        self._test_then_error(raise_value_error, "Expected error")

    def test_add_done_callback_simple(self):
        # [FUNCTION DOC] Checks a done callback does not run before completion and runs once a
        # result is set, using a flag updated by the callback.
        callback_result = False

        def callback(fut):
            # [FUNCTION DOC] HELPER: Callback that waits on the future and sets an enclosing
            # flag to true.
            nonlocal callback_result
            fut.wait()
            callback_result = True

        fut = Future[torch.Tensor]()
        fut.add_done_callback(callback)

        self.assertFalse(callback_result)
        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        self.assertTrue(callback_result)

    def test_add_done_callback_maintains_callback_order(self):
        # [FUNCTION DOC] Registers two done callbacks and checks they run in registration order,
        # so the last one wins the shared counter value of 2.
        callback_result = 0

        def callback_set1(fut):
            # [FUNCTION DOC] HELPER: Callback that waits on the future and sets the shared
            # counter to 1.
            nonlocal callback_result
            fut.wait()
            callback_result = 1

        def callback_set2(fut):
            # [FUNCTION DOC] HELPER: Callback that waits on the future and sets the shared
            # counter to 2.
            nonlocal callback_result
            fut.wait()
            callback_result = 2

        fut = Future[torch.Tensor]()
        fut.add_done_callback(callback_set1)
        fut.add_done_callback(callback_set2)

        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        # set2 called last, callback_result = 2
        self.assertEqual(callback_result, 2)

    def _test_add_done_callback_error_ignored(self, cb):
        # [FUNCTION DOC] HELPER: Registers the callback as a done callback on an integer future,
        # completes it with 5 and checks wait still returns 5, i.e. callback errors do not
        # propagate.
        fut = Future[int]()
        fut.add_done_callback(cb)

        fut.set_result(5)
        # error msg logged to stdout
        self.assertEqual(5, fut.wait())

    def test_add_done_callback_error_is_ignored(self):
        # [FUNCTION DOC] Checks a done callback that raises ValueError does not affect the
        # future.

        def raise_value_error(fut):
            # [FUNCTION DOC] HELPER: Callback that raises a ValueError (Expected error).
            raise ValueError("Expected error")

        self._test_add_done_callback_error_ignored(raise_value_error)

    def test_add_done_callback_no_arg_error_is_ignored(self):
        # [FUNCTION DOC] Checks a done callback with the wrong parameter count is ignored
        # without failing.

        def no_arg():
            # [FUNCTION DOC] HELPER: Callback with no parameters that returns True.
            return True

        # Adding another level of function indirection here on purpose.
        # Otherwise mypy will pick up on no_arg having an incompatible type and fail CI
        self._test_add_done_callback_error_ignored(no_arg)

    def test_interleaving_then_and_add_done_callback_maintains_callback_order(self):
        # [FUNCTION DOC] Interleaves done callbacks with a then callback and checks they run in
        # registration order: the then callback sees the counter set by the first done callback,
        # and the final counter is set by the last one.
        callback_result = 0

        def callback_set1(fut):
            # [FUNCTION DOC] HELPER: Callback that waits on the future and sets the shared
            # counter to 1.
            nonlocal callback_result
            fut.wait()
            callback_result = 1

        def callback_set2(fut):
            # [FUNCTION DOC] HELPER: Callback that waits on the future and sets the shared
            # counter to 2.
            nonlocal callback_result
            fut.wait()
            callback_result = 2

        def callback_then(fut):
            # [FUNCTION DOC] HELPER: Then callback returning the future's value plus the current
            # counter.
            nonlocal callback_result
            return fut.wait() + callback_result

        fut = Future[torch.Tensor]()
        fut.add_done_callback(callback_set1)
        then_fut = fut.then(callback_then)
        fut.add_done_callback(callback_set2)

        fut.set_result(torch.ones(2, 2))
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        # then_fut's callback is called with callback_result = 1
        self.assertEqual(then_fut.wait(), torch.ones(2, 2) + 1)
        # set2 called last, callback_result = 2
        self.assertEqual(callback_result, 2)

    def test_interleaving_then_and_add_done_callback_propagates_error(self):
        # [FUNCTION DOC] Checks that an error from a done callback is swallowed while the same
        # error from a then callback is raised when waiting on the chained future.
        def raise_value_error(fut):
            # [FUNCTION DOC] HELPER: Callback that raises a ValueError (Expected error).
            raise ValueError("Expected error")

        fut = Future[torch.Tensor]()
        then_fut = fut.then(raise_value_error)
        fut.add_done_callback(raise_value_error)
        fut.set_result(torch.ones(2, 2))

        # error from add_done_callback's callback is swallowed
        # error from then's callback is not
        self.assertEqual(fut.wait(), torch.ones(2, 2))
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            then_fut.wait()

    def test_collect_all(self):
        # [FUNCTION DOC] Checks collect_all returns a future that completes when all inputs do
        # (one set from a background thread) and yields the input futures in order.
        fut1 = Future[int]()
        fut2 = Future[int]()
        fut_all = torch.futures.collect_all([fut1, fut2])

        def slow_in_thread(fut, value):
            # [FUNCTION DOC] HELPER: Thread body that sleeps briefly then sets the given value
            # on the future.
            time.sleep(0.1)
            fut.set_result(value)

        t = threading.Thread(target=slow_in_thread, args=(fut1, 1))
        fut2.set_result(2)
        t.start()

        res = fut_all.wait()
        self.assertEqual(res[0].wait(), 1)
        self.assertEqual(res[1].wait(), 2)
        t.join()

    @unittest.skipIf(IS_WINDOWS, "TODO: need to fix this testcase for Windows")
    def test_wait_all(self):
        # [FUNCTION DOC] Skipped on Windows. Checks wait_all returns the list of results of
        # completed futures, and raises a RuntimeError when any input future carries an error.
        # Also prints the result.
        fut1 = Future[int]()
        fut2 = Future[int]()

        # No error version
        fut1.set_result(1)
        fut2.set_result(2)
        res = torch.futures.wait_all([fut1, fut2])
        print(res)
        self.assertEqual(res, [1, 2])

        # Version with an exception
        def raise_in_fut(fut):
            # [FUNCTION DOC] HELPER: Callback that raises a ValueError (Expected error).
            raise ValueError("Expected error")
        fut3 = fut1.then(raise_in_fut)
        with self.assertRaisesRegex(RuntimeError, "Expected error"):
            torch.futures.wait_all([fut3, fut2])

    def test_wait_none(self):
        # [FUNCTION DOC] Checks that passing None as a future to torch.jit.wait, wait_all or
        # collect_all raises a RuntimeError saying the future cannot be None.
        fut1 = Future[int]()
        with self.assertRaisesRegex(RuntimeError, "Future can't be None"):
            torch.jit.wait(None)
        with self.assertRaisesRegex(RuntimeError, "Future can't be None"):
            torch.futures.wait_all((None,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "Future can't be None"):
            torch.futures.collect_all((fut1, None,))  # type: ignore[arg-type]

if __name__ == '__main__':
    run_tests()
