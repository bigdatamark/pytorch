# [MODULE DOC] Leaf utility module for generating AOTAutograd runtime-wrapper
# source. Provides PySourceBuilder (indented source plus exec globals and a
# fresh-name counter), _compile_and_exec_source (compiles and execs generated
# source, logs it as a structured-trace artifact), GeneratedSource (record of one
# codegen'd function) and capture_generated_sources (thread-local capture sink).
# Import-time side effects: creates a module logger and a thread-local object.
"""
Generic codegen for AOTAutograd runtime wrappers.

PySourceBuilder accumulates indented Python source together with the globals the
generated code closes over and a fresh-name counter; _compile_and_exec_source is
the single chokepoint that turns generated source into a live function (logging it
as a structured-trace artifact for tlparse and giving it a stable traceback
filename). Every runtime-wrapper codegen routes through it -- the subclass wrappers
in subclass_codegen.py and the orchestration / alias / mutation epilogues in
runtime_wrappers.py -- so it knows nothing about any particular wrapper kind. This
module also hosts the optional thread-local capture sink that records each codegen'd
wrapper's source, which AOT-to-Python lowering (to_standalone_python.py) uses to
compose the wrappers into one standalone module. Kept as a leaf module (stdlib +
torch only) so it is safe to import anywhere.
"""

import contextlib
import functools
import logging
import threading
from collections.abc import Callable, Iterator

import torch
from torch.utils._indented_buffer import IndentedBuffer


log = logging.getLogger(__name__)


class PySourceBuilder(IndentedBuffer):
    # [CLASS DOC] Accumulates lines of generated Python source (via the
    # IndentedBuffer indentation machinery) plus state needed to run it: globals
    # (mapping of names to live objects the code closes over), a private
    # fresh-name counter, and optional function name / artifact name. If a
    # function name is given, a def header is emitted at construction and build()
    # becomes usable. Key methods: emit, fresh_name, add_global, bind, bind_value,
    # getvalue, build.
    """Builds indented Python source for compile/exec, along with the globals
    the generated code closes over and a monotonic fresh-name counter.

    Inherits the indentation primitive (``writeline``/``indent()``) from
    IndentedBuffer: body lines are written WITHOUT leading whitespace and
    indentation is managed by the ``indent()`` context manager so call sites
    read as plain code. This subclass adds the exec layer (globals binding,
    fresh-name generation, and ``build()`` via _compile_and_exec_source). Pass
    ``fn_name``/``artifact_name`` to emit a ``def`` header and enable
    ``build()``.
    """

    def __init__(
        self,
        fn_name: str | None = None,
        *,
        args: str = "args",
        artifact_name: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes an empty buffer with an empty globals mapping
        # and a zero name counter, storing fn_name and artifact_name. If fn_name is
        # given, writes a function header line taking the argument text args
        # (default a single varargs-like name "args") as the first line.
        super().__init__()
        self.globals: dict[str, object] = {}
        self._name_counter: int = 0
        self._fn_name = fn_name
        self._artifact_name = artifact_name
        if fn_name is not None:
            self.writeline(f"def {fn_name}({args}):")

    def emit(self, line: str, indent: int = 1) -> None:
        # [FUNCTION DOC] Appends line at the absolute indent level given (default 1),
        # temporarily overriding the buffer's current indent and restoring it
        # afterwards even on error. Mutates the buffer's line list only.
        """Append one line at an explicit absolute indent level.

        Convenient for recursive generators that thread an indent depth instead
        of nesting indent() context managers.
        """
        saved, self._indent = self._indent, indent
        try:
            self.writeline(line)
        finally:
            self._indent = saved

    def fresh_name(self, prefix: str) -> str:
        # [FUNCTION DOC] Returns prefix joined by an underscore to the current
        # counter value and increments the counter, so names are unique per builder.
        name = f"{prefix}_{self._name_counter}"
        self._name_counter += 1
        return name

    def add_global(self, name: str, value: object) -> str:
        # [FUNCTION DOC] Stores value in the globals mapping under name (silently
        # overwriting an existing entry) and returns name.
        """Bind a live object into the exec globals under ``name``, by reference."""
        self.globals[name] = value
        return name

    def bind(self, **values: object) -> None:
        # [FUNCTION DOC] Merges all keyword arguments into the globals mapping,
        # overwriting same-named entries. Returns nothing.
        """Bind live objects into the exec globals by keyword, by reference."""
        self.globals.update(values)

    def bind_value(self, prefix: str, value: object) -> str:
        # [FUNCTION DOC] Generates a fresh unique name from prefix, binds value to it
        # in the globals mapping, and returns the new name.
        """Bind a value under a fresh unique name and return that name."""
        return self.add_global(self.fresh_name(prefix), value)

    def getvalue(self) -> str:
        # [FUNCTION DOC] Returns the accumulated source as one string with lines
        # joined by newlines and no trailing newline. Raises AssertionError if any
        # buffered line is not a plain string (e.g. a deferred line object).
        # Join without a trailing newline (unlike IndentedBuffer.getvalue).
        if not all(isinstance(li, str) for li in self._lines):
            raise AssertionError(f"expected all lines to be str, got {self._lines}")
        return "\n".join(self._lines)  # type: ignore[arg-type]

    def build(
        self, *, wrapped_fn: Callable[..., object] | None = None
    ) -> Callable[..., object]:
        # [FUNCTION DOC] Compiles and execs the accumulated source with the builder's
        # globals and returns the generated function named by fn_name. If wrapped_fn
        # is given, wrapper metadata is copied from it onto the result. Raises
        # AssertionError if fn_name or artifact_name was not supplied at
        # construction. Side effects: logging and structured trace (see
        # _compile_and_exec_source) and capture-sink recording if active.
        if self._fn_name is None or self._artifact_name is None:
            raise AssertionError("build() requires fn_name and artifact_name")
        return _compile_and_exec_source(
            self.getvalue(),
            self.globals,
            self._fn_name,
            self._artifact_name,
            wrapped_fn=wrapped_fn,
        )


# Optional sink for the source of every runtime-wrapper function codegen'd via
# ``_compile_and_exec_source``. When a sink is installed (see
# ``capture_generated_sources``), each codegen'd wrapper appends a
# ``GeneratedSource`` to it; this lets AOT-to-Python lowering compose the wrappers
# into one standalone module. Thread-local (NOT a process global) so a concurrent
# compile on another thread cannot splice its wrappers into this capture -- and,
# conversely, so an unrelated concurrent compile on another thread is never mistaken for
# offloaded capture work and aborted (``_compile_and_exec_source`` runs on EVERY
# AOTAutograd compile, not just this one). Absent (zero overhead, no behavior change)
# during ordinary compilation. Mirrors the threading.local used by
# _saved_tensor_hook_context in graph_compile.py.
_capture_tls = threading.local()


def _current_capture_sink() -> "list[GeneratedSource] | None":
    # [FUNCTION DOC] HELPER: Returns the capture list installed for the current
    # thread, or None when no capture is active.
    return getattr(_capture_tls, "sink", None)


class GeneratedSource:
    # [CLASS DOC] Plain record of one codegen'd wrapper function. Attributes:
    # artifact_name, fn_name, source text, globals_dict (pre-exec snapshot of the
    # closure globals, without interpreter builtins), fn (the live function), and
    # origin_id (identity of the TracingContext at codegen time, or None). Created
    # only by _compile_and_exec_source when a capture sink is active.
    """One codegen'd runtime-wrapper function: its source, the exec'd function
    object, and the globals it closes over (which include the inner ``compiled_fn``
    it chains to, plus any baked metadata). The function object lets a composer wire
    cross-wrapper references by identity. Recorded by ``_compile_and_exec_source``
    when capture is active.

    ``globals_dict`` is a PRE-EXEC snapshot of the declared closure globals: it does
    NOT contain the interpreter ``__builtins__`` that exec() would otherwise inject,
    so the standalone composer can reconstruct every entry as source without having
    to special-case ``__builtins__``."""

    def __init__(
        self,
        artifact_name: str,
        fn_name: str,
        source: str,
        globals_dict: dict[str, object],
        fn: object,
        origin_id: int | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores each argument verbatim as an attribute of the same
        # name. No validation and no side effects.
        self.artifact_name = artifact_name
        self.fn_name = fn_name
        self.source = source
        self.globals_dict = globals_dict
        self.fn = fn
        # Identity of the TracingContext this wrapper was codegen'd under. The capture
        # sink is duration-scoped over one inductor compile, so a re-entrant on-thread
        # lowering that codegen's into THIS sink during that window would append its
        # wrappers too; the composer filters by this id, which separates such a foreign
        # lowering when it ran under a DISTINCT TracingContext. A same-context re-entrant
        # lowering shares this id and is caught instead by the composer's
        # orchestration-count guard.
        self.origin_id = origin_id


@contextlib.contextmanager
def capture_generated_sources(into: "list[GeneratedSource]") -> "Iterator[None]":
    # [FUNCTION DOC] Context manager. While active, installs into as the
    # thread-local capture sink so each wrapper codegen'd on this thread appends a
    # GeneratedSource to it in codegen order. On exit (even on error) restores the
    # previous sink, so nesting works. Wrappers codegen'd on other threads are not
    # captured.
    """Within this context, record every codegen'd runtime-wrapper function's source
    into ``into`` (in codegen order). A no-op when not entered.

    THREADING: the sink is thread-local, so every wrapper captured into ``into`` must be
    codegen'd ON THIS THREAD. There is deliberately NO cross-thread tripwire here: a
    process-global owner check would abort an unrelated concurrent compile on another
    thread, since ``_compile_and_exec_source`` runs on every AOTAutograd compile and is not
    serialized against this capture. If a future change offloads wrapper codegen to a
    worker thread, those wrappers are simply not captured; a forward that loses its
    orchestration wrapper this way is rejected by the composer (it requires exactly one),
    so the common case fails loudly rather than emitting an empty module.
    """
    prev = _current_capture_sink()
    _capture_tls.sink = into
    try:
        yield
    finally:
        _capture_tls.sink = prev


def _compile_and_exec_source(
    source: str,
    globals_dict: dict[str, object],
    fn_name: str,
    artifact_name: str,
    wrapped_fn: Callable[..., object] | None = None,
) -> Callable[..., object]:
    # [FUNCTION DOC] HELPER: Single chokepoint turning generated source into a live
    # function. Logs the source at debug level and as a structured-trace "artifact"
    # named artifact_name, compiles it under a synthetic filename inside this file's
    # path (so dynamo skips it), execs it with globals_dict (which exec mutates by
    # injecting builtins) and returns the entry named fn_name. If wrapped_fn is
    # given, copies its wrapper metadata onto the result. If a capture sink is
    # active, appends a GeneratedSource with a pre-exec globals snapshot and the
    # current TracingContext identity. Raises whatever compile/exec raise (e.g.
    # SyntaxError) or a lookup error if fn_name is not defined by the source.
    """Compile generated source, exec it, and return the named function.

    If wrapped_fn is provided, applies functools.update_wrapper so that
    __wrapped__ and __dict__ (e.g. _fx_graph_cache_key) propagate to the
    generated function.
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Generated %s:\n%s", artifact_name, source)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: source,
    )

    # Use a path under torch/_functorch/ so the code object is recognized by
    # dynamo's MOD_SKIPLIST. The eval frame hook stays active during the entire
    # torch.compile(fn)(*args) call (to handle graph breaks and resume functions),
    # so codegen'd functions called during backward get intercepted even though
    # no tracing is active. A real path makes them skip automatically.
    code = compile(source, f"{__file__}:codegen({artifact_name})", "exec")
    local_dict: dict[str, object] = {}
    # exec() mutates ``globals_dict`` in place, injecting ``__builtins__`` (and any
    # name the source binds at module scope). When capturing, snapshot the declared
    # closed-over names BEFORE that happens so the captured GeneratedSource holds only
    # the intended closure globals -- not the post-exec dict with the interpreter's
    # ``__builtins__`` -- which is what the standalone composer reconstructs as source.
    sink = _current_capture_sink()
    captured_globals = dict(globals_dict) if sink is not None else globals_dict
    exec(code, globals_dict, local_dict)
    fn = local_dict[fn_name]
    if wrapped_fn is not None:
        functools.update_wrapper(fn, wrapped_fn)  # type: ignore[arg-type]

    if sink is not None:
        # Tag with the current TracingContext identity so the composer can drop any
        # wrapper a re-entrant lowering appended during the capture window (see the
        # origin_id note on GeneratedSource).
        ctx = torch._guards.TracingContext.try_get()
        origin_id = id(ctx) if ctx is not None else None
        sink.append(
            GeneratedSource(
                artifact_name, fn_name, source, captured_globals, fn, origin_id
            )
        )

    return fn  # type: ignore[return-value]
