# [MODULE DOC] Utilities to run awaitables synchronously from sync code, even when an
# event loop is already running or closed in the current thread. Provides await_sync,
# the get_loop context manager, and private helpers that create loops which track
# their tasks and cancel leftovers on exit. Defines task-factory typing aliases
# depending on Python version. No other import-time side effects.
import asyncio
import sys
import weakref
from asyncio import AbstractEventLoop, Future
from collections.abc import Awaitable, Callable, Coroutine, Generator, Iterator
from contextlib import contextmanager, ExitStack
from contextvars import Context
from typing import Any, Protocol, TypeVar

from torch.utils._ordered_set import OrderedSet


T = TypeVar("T")
TCoro = Generator[Any, None, T]

if sys.version_info >= (3, 11):

    class TaskFactory(Protocol):
        # [FUNCTION DOC] HELPER: Protocol signature for an asyncio task factory taking
        # a loop, a coroutine and an optional context, returning a future.
        def __call__(
            self,
            __loop: AbstractEventLoop,
            __factory: Coroutine[None, None, object] | Generator[None, None, object],
            __context: Context | None = None,
            /,
        ) -> asyncio.futures.Future[object]: ...

    TaskFactoryType = TaskFactory
else:
    TaskFactoryType = Callable[[AbstractEventLoop, Generator[TCoro, None, T]], Future]  # type: ignore[valid-type]


def await_sync(awaitable: Awaitable[T]) -> T:
    # [FUNCTION DOC] Runs the awaitable to completion on a loop obtained from get_loop
    # and returns its result; exceptions from the awaitable propagate. May create and
    # close a temporary event loop and temporarily replace the thread current loop.
    with get_loop() as loop:
        return loop.run_until_complete(awaitable)


@contextmanager
def get_loop(
    always_create_new_loop: bool = False,
) -> Iterator[AbstractEventLoop]:
    # [FUNCTION DOC] Context manager yielding a usable event loop. If the thread has no
    # current loop, a fresh patched loop is made and closed afterward. If the current
    # loop is running, the running-loop marker is cleared and a new loop (inheriting
    # the task factory) is used so run_until_complete works; closed loops or
    # always_create_new_loop=True also get a new loop. Original loop is restored on exit.
    # Raises any RuntimeError from get_event_loop other than the no-current-loop one.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as re:
        if "There is no current event loop in thread" in str(re):
            with _new_loop() as loop:
                yield loop
            return
        else:
            raise

    @contextmanager
    def _restore_loop(
        loop: asyncio.AbstractEventLoop,
    ) -> Iterator[None]:
        # [FUNCTION DOC] HELPER: On exit (always), sets the given loop as the thread
        # current event loop.
        try:
            yield
        finally:
            asyncio.set_event_loop(loop)

    @contextmanager
    def _restore_running_loop() -> Iterator[None]:
        # [FUNCTION DOC] HELPER: Clears the asyncio running-loop marker for the duration
        # of the block and restores the previous marker on exit (uses private asyncio API).
        loop_from_events = asyncio.events._get_running_loop()
        asyncio.events._set_running_loop(None)
        try:
            yield
        finally:
            asyncio.events._set_running_loop(loop_from_events)

    with ExitStack() as stack:
        if loop.is_running():
            stack.enter_context(_restore_running_loop())
            stack.enter_context(_restore_loop(loop=loop))
            loop = stack.enter_context(_new_loop(loop.get_task_factory()))  # type: ignore[arg-type]
        elif loop.is_closed():
            loop = stack.enter_context(_new_loop())  # type: ignore[arg-type]
        elif always_create_new_loop:
            stack.enter_context(_restore_loop(loop=loop))
            loop = stack.enter_context(_new_loop())  # type: ignore[arg-type]
        yield loop


@contextmanager
def _new_loop(
    task_factory: TaskFactoryType | None = None,
) -> Iterator[asyncio.AbstractEventLoop]:
    # [FUNCTION DOC] HELPER: Creates a new event loop patched to track its tasks,
    # optionally installs a task factory, and makes it the current loop. On exit it
    # cancels all unfinished tasks, unsets the current loop and closes the loop.
    loop = asyncio.new_event_loop()
    tasks = _patch_loop(loop)

    if task_factory:
        # pyre-ignore[6]
        loop.set_task_factory(task_factory)  # type: ignore[arg-type]

    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        try:
            _cancel_all_tasks(loop, tasks)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _cancel_all_tasks(
    loop: AbstractEventLoop,
    tasks: OrderedSet[Future],  # type: ignore[type-arg]
) -> None:
    # [FUNCTION DOC] HELPER: Cancels every not-yet-done task in the set, waits for them
    # on the loop (collecting exceptions), then reports any task exception that is not a
    # cancellation via the loop exception handler. Returns nothing.
    to_cancel = [task for task in tasks if not task.done()]

    if not to_cancel:
        return

    # pyre-fixme[1001]: Awaitable assigned to `task` is never awaited.
    for task in to_cancel:
        task.cancel()

    # pyrefly: ignore [bad-argument-type]
    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def _patch_loop(loop: AbstractEventLoop) -> OrderedSet[Future]:  # type: ignore[type-arg]
    # [FUNCTION DOC] HELPER: Installs on the loop a task factory that records every
    # created task in a weak set, and overrides the loop set_task_factory and
    # get_task_factory so user factories are stored and delegated to by the tracking
    # factory instead of replacing it. Mutates the loop. Returns the weak set of tasks
    # (typed as an ordered set).
    tasks: weakref.WeakSet[Future] = weakref.WeakSet()  # type: ignore[type-arg]

    task_factories: list[TaskFactoryType | None] = [None]

    def _set_task_factory(factory: TaskFactoryType | None) -> None:
        # [FUNCTION DOC] HELPER: Stores the user task factory (or none) in shared state.
        task_factories[0] = factory

    def _get_task_factory() -> TaskFactoryType | None:
        # [FUNCTION DOC] HELPER: Returns the currently stored user task factory or none.
        return task_factories[0]

    def _safe_task_factory(
        loop: AbstractEventLoop,
        coro: TCoro,  # type: ignore[type-arg]
        *,
        context: Context | None = None,
    ) -> asyncio.Future:  # type: ignore[valid-type, type-arg]
        # [FUNCTION DOC] HELPER: Creates a task for the coroutine, using the stored user
        # factory if any or a plain asyncio Task otherwise (passing context on 3.11+),
        # drops the last frame of the source traceback for plain tasks, records the task
        # in the tracking set and returns it.
        task_factory = task_factories[0]
        if task_factory is None:
            if sys.version_info >= (3, 11):
                # pyrefly: ignore [bad-argument-type]
                task = asyncio.Task(coro, loop=loop, context=context)
            else:
                task = asyncio.Task(coro, loop=loop)
            # pyre-ignore[16]: `Task` has no attribute `_source_traceback`.
            if task._source_traceback:  # type: ignore[attr-defined]
                del task._source_traceback[  # type: ignore[attr-defined]
                    -1
                ]  # pragma: no cover  # type: ignore[attr-defined]
        else:
            if sys.version_info >= (3, 11):
                task = task_factory(loop, coro, context=context)  # type: ignore[arg-type, call-arg, assignment]
            else:
                task = task_factory(loop, coro)  # type: ignore[arg-type]
        #  `Union[Task[Any], Future[Any]]`.
        tasks.add(task)
        return task

    # pyre-ignore[6]
    loop.set_task_factory(_safe_task_factory)  # type: ignore[method-assign, arg-type]
    # pyre-ignore[8]
    loop.set_task_factory = _set_task_factory  # type: ignore[method-assign, assignment]
    # pyre-ignore[8]
    loop.get_task_factory = _get_task_factory  # type: ignore[method-assign, assignment]

    return tasks  # type: ignore[return-value]
