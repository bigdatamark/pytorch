# mypy: allow-untyped-defs
# [MODULE DOC] Queue classes whose connections serialize objects with ForkingPickler so the
# custom tensor reducers (shared memory) are used. Contents: ConnectionWrapper, Queue and
# SimpleQueue. No import-time side effects.
import io
import multiprocessing.queues
import pickle
from multiprocessing.reduction import ForkingPickler


class ConnectionWrapper:
    # [CLASS DOC] Proxy around a multiprocessing connection. State: conn, the wrapped connection.
    # Overrides send and recv to use ForkingPickler-based serialization; every other attribute
    # is forwarded to the wrapped connection.
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler for object serialization."""

    def __init__(self, conn):
        # [FUNCTION DOC] Stores the connection to wrap.
        self.conn = conn

    def send(self, obj):
        # [FUNCTION DOC] Pickles obj with ForkingPickler at the highest protocol (so registered
        # reducers apply) and sends the bytes over the wrapped connection.
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        # [FUNCTION DOC] Receives bytes from the wrapped connection and unpickles them, returning
        # the object. Blocks until data arrives; unpickling errors propagate.
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        # [FUNCTION DOC] Forwards unknown attribute lookups to the wrapped connection. Raises
        # AttributeError if the wrapper has no connection yet (guards recursion during unpickling).
        if "conn" in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'conn'")


class Queue(multiprocessing.queues.Queue):
    # [CLASS DOC] Standard multiprocessing Queue whose reader and writer connections are wrapped
    # so objects (including tensors) travel through ForkingPickler; its send and recv hooks are
    # rebound to the wrappers.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Builds the base queue with the given arguments, then wraps its reader and
        # writer in ConnectionWrapper and points the internal send/recv hooks at them.
        super().__init__(*args, **kwargs)
        self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
        self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    # [CLASS DOC] Standard multiprocessing SimpleQueue whose reader and writer are wrapped in
    # ConnectionWrapper (once) before the base class binds its put/get methods.
    def _make_methods(self):
        # [FUNCTION DOC] HELPER: Wraps reader and writer in ConnectionWrapper if not already
        # wrapped, then lets the base class create its methods from them.
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
            self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        super()._make_methods()  # type: ignore[misc]
