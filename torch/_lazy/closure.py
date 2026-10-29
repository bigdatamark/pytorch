# [MODULE DOC] Step-closure machinery for lazy tensors: closures registered with add_step_closure are stored on the
# per-device DeviceContext and executed by run_step_closures after a mark step, synchronously via ClosureHandler or on a
# background thread via AsyncClosureHandler.
import os
import threading
from collections.abc import Callable, Iterable
from queue import Empty as EmptyQueue, Queue
from typing import overload
from typing_extensions import TypeVarTuple, Unpack

from torch._lazy.device_context import DeviceContext, get_device_context


_Ts = TypeVarTuple("_Ts")


class ClosureHandler:
    # [CLASS DOC] Synchronous executor of step closures; stateless. A handler is called with an iterable of
    # zero-argument callables and runs each in order on the calling thread.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: No-op constructor.
        pass

    def run(self, closure: Callable[[], object]) -> None:
        # [FUNCTION DOC] Calls the given zero-argument closure immediately, ignoring its return value.
        """Run closure function

        Args:
        closure: callable function to run
        """
        closure()

    def __call__(self, closures: Iterable[Callable[[], object]]) -> None:
        # [FUNCTION DOC] Runs every closure in the iterable in order via run.
        for closure in closures:
            self.run(closure)


class AsyncClosureHandler(ClosureHandler):
    # [CLASS DOC] Runs closures on a single background worker thread fed by a bounded queue. State: closure queue
    # (size from env LTC_MAX_ASYNC_QUEUE, else max_queue_size), an exception queue, a lock, a finished event, and the
    # thread handle. run blocks when the queue is full; the worker exits after 3 seconds of idle time or on the first
    # closure exception, which is stored and re-raised (as RuntimeError) on the next run.
    """Handler for Asynchronous Step Closures
    Args:
        max_queue_size: The maximum length of the closure queue after which
        the training loop will block until closures are evaluated.
        By default, a reasonable limit of a maximum of 100 on the queue.
        This value can be set using the `XLA_MAX_ASYNC_QUEUE` environment
        variable.
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        # [FUNCTION DOC] Creates the bounded closure queue, the exception queue, the lock, the finished event and an unset
        # event-loop thread handle. The queue size is read from the LTC_MAX_ASYNC_QUEUE environment variable if set.
        super().__init__()
        self._closure_queue: Queue[Callable[[], object]] = Queue(
            int(os.environ.get("LTC_MAX_ASYNC_QUEUE", max_queue_size))
        )
        self._closure_exception: Queue[Exception] = Queue()
        self._closure_lock = threading.Lock()
        self._closure_event_loop_finished = threading.Event()
        self._closure_event_loop: threading.Thread | None = None

    def start_event_loop(self) -> None:
        # [FUNCTION DOC] Starts the worker thread if none is recorded; does nothing otherwise. Side effect: spawns a thread.
        """Start closure event loop if not started"""
        if self._closure_event_loop is None:

            def event_loop() -> None:
                # [FUNCTION DOC] HELPER: Worker loop. Takes closures from the queue (waiting up to 3 seconds), runs them and marks
                # them done. On timeout with an empty queue (checked under the lock) it sets the finished event and returns. Any
                # exception from a closure is put on the exception queue and the loop ends.
                # Run loop until closure event is set and closure queue is empty
                while True:
                    try:
                        closure = self._closure_queue.get(block=True, timeout=3)
                        closure()
                        self._closure_queue.task_done()
                    except EmptyQueue:
                        with self._closure_lock:
                            if self._closure_queue.empty():
                                self._closure_event_loop_finished.set()
                                return
                    except Exception as e:
                        self._closure_exception.put(e)
                        return

            self._closure_event_loop = threading.Thread(target=event_loop)
            self._closure_event_loop.start()

    def run(self, closure: Callable[[], object]) -> None:
        # [FUNCTION DOC] Enqueues the closure (blocking if the queue is full) under the lock. If the worker thread is missing
        # or dead, checks the exception queue: a stored exception raises RuntimeError chained to it; otherwise restarts the
        # worker thread.
        with self._closure_lock:
            self._closure_queue.put(closure, block=True)
            if (
                self._closure_event_loop is None
                or not self._closure_event_loop.is_alive()
            ):
                try:
                    e = self._closure_exception.get(block=False)
                    raise RuntimeError(
                        "Cannot run asynchronous closure due to previously raised exception"
                    ) from e
                except EmptyQueue:
                    self._closure_event_loop = None
                    self.start_event_loop()


@overload
def add_step_closure(
    closure: Callable[[], object],
    args: tuple[()] = (),
    run_async: bool = False,
) -> None: ...
@overload
def add_step_closure(
    closure: Callable[[Unpack[_Ts]], object],
    args: tuple[Unpack[_Ts]],
    run_async: bool = False,
) -> None: ...
def add_step_closure(
    closure: Callable[..., object],
    args: tuple[object, ...] = (),
    run_async: bool = False,
) -> None:
    # [FUNCTION DOC] Registers a closure and its arguments on the current device context to run after the next step
    # barrier. Uses the async list if run_async is true, else the sync list, creating the list on the context as needed.
    # Closures run in queued order. Returns None.
    """Adds a closure to the list of the ones to be run at the end of the step.
    Many times during model training there is the need to print/report (print to
    console, post to tensorboard, etc...) information which require the content of
    intermediary tensors to be inspected.
    Inspecting different tensors content in different points of the model code
    requires many executions and typically causes performance issues.
    Adding a step closure will ensure that it will be run after the barrier, when
    all the live tensors will be already materialized to device data.
    Live tensors which will include the ones captured by the closure arguments.
    So using `add_step_closure()` will ensure a single execution will be
    performed, even when multiple closures are queued, requiring multiple tensors
    to be inspected.
    Step closures will be run sequentially in the order they have been queued.
    Note that even though using this API the execution will be optimized, it is
    advised to throttle the printing/reporting events once every N steps.
    Args:
      closure (callable): The function to be called.
      args (tuple): The arguments to be passed to the closure.
      run_async: If True, run the closure asynchronously.
    """
    devctx = get_device_context()
    closures_type = "async_step_closures" if run_async else "step_closures"
    step_closures: list[Callable[[], object]] | None = getattr(
        devctx, closures_type, None
    )
    if step_closures is None:
        step_closures = []
        setattr(devctx, closures_type, step_closures)
    step_closures.append(lambda a=args: closure(*a))


def run_step_closures() -> DeviceContext:
    # [FUNCTION DOC] Executes and clears the pending async closures (via a lazily created AsyncClosureHandler) and sync
    # closures (via a lazily created ClosureHandler) stored on the current device context. Returns the device context.
    # Side effects: mutates the context's closure lists and handler attributes; closures run.
    devctx = get_device_context()
    async_step_closures = getattr(devctx, "async_step_closures", None)
    if async_step_closures is not None:
        devctx.async_step_closures = []  # type: ignore[attr-defined]
        async_closure_handler = getattr(devctx, "async_closure_handler", None)
        if async_closure_handler is None:
            async_closure_handler = AsyncClosureHandler()
            devctx.async_closure_handler = async_closure_handler  # type: ignore[attr-defined]
        async_closure_handler(async_step_closures)

    step_closures = getattr(devctx, "step_closures", None)
    if step_closures is not None:
        devctx.step_closures = []  # type: ignore[attr-defined]
        closure_handler = getattr(devctx, "closure_handler", None)
        if closure_handler is None:
            closure_handler = ClosureHandler()
            devctx.closure_handler = closure_handler  # type: ignore[attr-defined]
        closure_handler(step_closures)
    return devctx
