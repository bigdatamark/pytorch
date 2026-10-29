# [MODULE DOC] Pool variant that moves tensors between workers through shared memory by using
# this package's SimpleQueue, and workers that run a garbage collection after finishing.
# Contents: clean_worker and Pool. No import-time side effects.
import multiprocessing.pool
import multiprocessing.util as util

from .queue import SimpleQueue


def clean_worker(*args, **kwargs):
    # [FUNCTION DOC] Worker entry point: runs the standard pool worker loop with the given
    # arguments, then forces a garbage collection so destructors of objects held by the worker run
    # before the process ends.
    import gc

    multiprocessing.pool.worker(*args, **kwargs)
    # Regular multiprocessing workers don't fully clean up after themselves,
    # so we have to explicitly trigger garbage collection to make sure that all
    # destructors are called...
    gc.collect()


class Pool(multiprocessing.pool.Pool):
    # [CLASS DOC] Process pool using this package's SimpleQueue for task and result queues and
    # clean_worker as the worker target. Overrides internal queue setup and worker repopulation
    # of the standard Pool; relies on private attributes of the base class.
    """Pool implementation which uses our version of SimpleQueue.

    This lets us pass tensors in shared memory across processes instead of
    serializing the underlying data.
    """

    def _setup_queues(self):
        # [FUNCTION DOC] HELPER: Creates the input and output queues as the shared-memory-aware
        # SimpleQueue and sets the fast put/get hooks to the writer's send and the reader's recv.
        self._inqueue = SimpleQueue()
        self._outqueue = SimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv

    def _repopulate_pool(self):
        # [FUNCTION DOC] HELPER: Starts enough new daemon worker processes to bring the pool back
        # to the configured size, each running clean_worker with the queues, initializer,
        # max tasks per child and (if the base class has it) the exception wrapper. Workers are
        # renamed from Process to PoolWorker. Side effect: appends to the pool list and starts processes.
        """Increase the number of pool processes to the specified number.

        Bring the number of pool processes up to the specified number, for use after
        reaping workers which have exited.
        """
        for _ in range(self._processes - len(self._pool)):
            # changed worker -> clean_worker
            args = (
                self._inqueue,
                self._outqueue,
                self._initializer,
                self._initargs,
                self._maxtasksperchild,
            )
            if hasattr(self, "_wrap_exception"):
                args += (self._wrap_exception,)
            w = self.Process(target=clean_worker, args=args)
            self._pool.append(w)
            w.name = w.name.replace("Process", "PoolWorker")
            w.daemon = True
            w.start()
            util.debug("added worker")
