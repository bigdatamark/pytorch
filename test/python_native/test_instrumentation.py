# Owner(s): ["module: dsl-native-ops"]
# [MODULE DOC] Tests for torch._native.instrumentation, the logging and structured-trace
# instrumentation of native DSL compile and launch sites. Defines fake Triton, CuTeDSL and Helion
# kernel stand-ins (no GPU needed), test cases checking compiled versus cache_hit reporting, error
# re-raising and attribute passthrough, a structured-trace/tlparse test case, and an AST based
# coverage guard over torch/_native/ops requiring every DSL compile site to carry instrumentation.
# Import-time: detects whether the tlparse binary exists.

import ast
import json
import logging
import os
import shutil
import subprocess
import tempfile
import unittest
from collections import namedtuple

from torch._logging._internal import TorchLogsFormatter, trace_log
from torch._native.instrumentation import (
    CompileEvent,
    instrument_cutedsl_compile,
    instrument_helion_kernel,
    instrument_triton_kernel,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.logging_utils import log_settings, preserve_log_state


# No shared tlparse harness exists in torch (see test/dynamo/test_structured_trace.py,
# which defines these locally too), so mirror its minimal pattern here.
HAS_TLPARSE = shutil.which("tlparse") is not None
requires_tlparse = unittest.skipUnless(HAS_TLPARSE, "requires tlparse")


# Mirror of torch._vendor.quack.cache.CacheInfo. Defined locally so the test
# doesn't import quack (which pulls in cutlass, absent on CPU-only builds).
_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


class _FakeJITFunction:
    # [CLASS DOC] HELPER: Fake of a Triton jit function. Has a per-device cache table with one
    # device (cuda:0) holding a dict of compiled variants, and a flag making launches fail.
    # Indexing with a grid returns a launcher that adds a new variant to the cache when not yet
    # present.
    """Stand-in for a ``@triton.jit`` kernel. GPU- and Triton-free.

    ``instrument_triton_kernel`` watches ``device_caches[dev][0]`` (the dict
    of compiled variants) and launches via ``kernel[grid](*args)``. We model
    one fake device whose cache grows on each distinct ``variant`` kwarg (a
    fresh Triton compile / miss); a repeated variant leaves it unchanged (a
    cache hit).
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Create an empty compiled-variant cache, expose it as the device
        # caches mapping for cuda:0, and set raise_on_launch to False.
        # defaultdict-like: one device "cuda:0", value is a (cache_dict, ...) tuple.
        self._cache: dict = {}
        self.device_caches = {"cuda:0": (self._cache, None)}
        self.raise_on_launch = False

    def __getitem__(self, grid):
        # [FUNCTION DOC] HELPER: Mimic launch by grid subscripting. Returns a launcher function for
        # the given grid (the grid is ignored).
        def launcher(*args, variant="v0", **kwargs):
            # [FUNCTION DOC] HELPER: Fake launch. Raises RuntimeError (message kaboom) if
            # raise_on_launch is set; otherwise records a new placeholder object in the cache for
            # the variant keyword (default v0) if absent and returns the string launched.
            if self.raise_on_launch:
                raise RuntimeError("kaboom")
            if variant not in self._cache:
                self._cache[variant] = object()
            return "launched"

        return launcher


class _FakeJitCache:
    # [CLASS DOC] HELPER: Fake of a cached CuTeDSL compile function. Keeps a store keyed by call
    # arguments, hit and miss counters, and a flag making calls fail. Calling returns the cached
    # object, counting a hit or a miss; cache_info reports the counters like a functools cache.
    """Stand-in for a ``@jit_cache``-decorated compile function.

    Mimics the bits ``instrument_cutedsl_compile`` observes: a ``cache_info()``
    whose ``misses`` advances on a cold key, plus a controllable wall time
    via the compiled callable. Keeps the test GPU- and CuTeDSL-free.
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialize an empty store, zero hit and miss counters and the
        # raise_on_call flag set to False.
        self._cache: dict = {}
        self.hits = 0
        self.misses = 0
        self.raise_on_call = False

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake compile call. Raises RuntimeError (message boom) if
        # raise_on_call is set. Builds a key from positional args and sorted keyword items;
        # increments hits if cached, else increments misses and stores a new object. Returns the
        # stored object for the key.
        if self.raise_on_call:
            raise RuntimeError("boom")
        key = args + tuple(sorted(kwargs.items()))
        if key in self._cache:
            self.hits += 1
        else:
            self.misses += 1
            self._cache[key] = object()
        return self._cache[key]

    def cache_info(self):
        # [FUNCTION DOC] HELPER: Return a named tuple of hits, misses, maxsize (None) and the
        # current store size.
        return _CacheInfo(
            hits=self.hits,
            misses=self.misses,
            maxsize=None,
            currsize=len(self._cache),
        )


class _FakeHelionBoundKernel:
    # [CLASS DOC] HELPER: Fake of a Helion bound kernel; only holds a compile cache attribute
    # containing one entry.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Create the compile cache mapping with a single placeholder entry.
        self._compile_cache = {"config": object()}


class _FakeHelionKernel:
    # [CLASS DOC] HELPER: Fake of a Helion kernel. Keeps a mapping of variant to bound kernel, a
    # flag making calls fail, and a cache key string. Calling it adds a bound kernel for the
    # variant if not yet present, which is how the instrumentation sees a compile.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Start with no bound kernels, raise_on_call False and the cache key
        # set to the string helion-cache.
        self._bound_kernels = {}
        self.raise_on_call = False
        self.cache_key = "helion-cache"

    def __call__(self, variant="v0"):
        # [FUNCTION DOC] HELPER: Fake kernel call with an optional variant name (default v0).
        # Raises RuntimeError (message helion boom) if raise_on_call is set; otherwise adds a bound
        # kernel for the variant if absent and returns the string launched.
        if self.raise_on_call:
            raise RuntimeError("helion boom")
        self._bound_kernels.setdefault(variant, _FakeHelionBoundKernel())
        return "launched"


class _CapturingHandler(logging.Handler):
    # [CLASS DOC] HELPER: Logging handler that stores every emitted record in a list attribute
    # (records) instead of writing it anywhere.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialize the base handler and an empty list of records.
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        # [FUNCTION DOC] HELPER: Append the received log record to the records list.
        self.records.append(record)


# Both sinks are gated on this off-by-default artifact, so tests asserting on
# emitted output have to turn it on first.
_ARTIFACT = "native_dsl_compile"
_ARTIFACT_LOG_QNAME = f"torch._native.instrumentation.__{_ARTIFACT}"


class _LoggerCaptureTest(TestCase):
    # [CLASS DOC] HELPER: Base test case that enables the native DSL compile artifact logging and
    # attaches a capturing handler to its logger, with propagation turned off to keep output off
    # stderr. Exposes the captured formatted messages through a property.
    """Captures the artifact logger so tests can assert on emitted lines."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Enable the artifact via log settings, get the artifact logger,
        # remember and disable its propagate flag, and attach a new capturing handler.
        super().setUp()
        self._log_settings = log_settings(f"+{_ARTIFACT}")
        self.log = logging.getLogger(_ARTIFACT_LOG_QNAME)
        self._orig_propagate = self.log.propagate
        self.log.propagate = False  # keep the lines off stderr
        self.handler = _CapturingHandler()
        self.log.addHandler(self.handler)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Detach the handler, restore the propagate flag, close the log
        # settings context and call the parent tearDown.
        self.log.removeHandler(self.handler)
        self.log.propagate = self._orig_propagate
        self._log_settings.close()
        super().tearDown()

    @property
    def messages(self):
        # [FUNCTION DOC] HELPER: Property returning the list of formatted message strings of all
        # records captured so far.
        return [r.getMessage() for r in self.handler.records]


class TestInstrumentation(_LoggerCaptureTest):
    # [CLASS DOC] Tests for the CuTeDSL compile instrumentation wrapper using the fake cached
    # compile function. Checks log text for first call versus repeat call, per-key compiles, custom
    # key function use, error reporting with re-raise, cache attribute forwarding, plain callables,
    # no work when the artifact is disabled, and the event JSON round trip.
    def test_first_call_reports_compiled(self):
        # [FUNCTION DOC] First call through the wrapper logs exactly one message naming the op, the
        # cutedsl tag, compiled and a miss count of one.
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)

        self.assertEqual(len(self.messages), 1)
        msg = self.messages[0]
        self.assertIn("aten::topk", msg)
        self.assertIn("[cutedsl]", msg)
        self.assertIn("compiled", msg)
        self.assertIn("misses=1", msg)

    def test_second_call_reports_cache_hit(self):
        # [FUNCTION DOC] Two identical calls log two messages: compiled first, then cache_hit with
        # one hit.
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)
        compile_fn(256, 64)

        self.assertEqual(len(self.messages), 2)
        self.assertIn("compiled", self.messages[0])
        self.assertIn("cache_hit", self.messages[1])
        self.assertIn("hits=1", self.messages[1])

    def test_distinct_keys_each_compile(self):
        # [FUNCTION DOC] Calls with two distinct argument sets give two misses, and every logged
        # message reports compiled.
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        compile_fn(256, 64)
        compile_fn(512, 128)

        self.assertEqual(fake.misses, 2)
        for msg in self.messages:
            self.assertIn("compiled", msg)

    def test_key_fn_used_in_log(self):
        # [FUNCTION DOC] A key function passed to the wrapper builds the key text, which appears in
        # the first log message.
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile(
            "aten::topk", key_fn=lambda N, K: f"radix N={N} K={K}"
        )(fake)

        compile_fn(256, 64)

        self.assertIn("radix N=256 K=64", self.messages[0])

    def test_error_is_reported_and_reraised(self):
        # [FUNCTION DOC] When the wrapped function raises, the wrapper re-raises RuntimeError and
        # logs one message containing the word error.
        fake = _FakeJitCache()
        fake.raise_on_call = True
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        with self.assertRaises(RuntimeError):
            compile_fn(256, 64)

        self.assertEqual(len(self.messages), 1)
        self.assertIn("error", self.messages[0])

    def test_cache_attrs_forwarded(self):
        # [FUNCTION DOC] Asserts the wrapper still exposes cache_info and its reported misses start
        # at zero.
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk")(fake)

        # jit_cache exposes cache_info / cache_clear; the wrapper must keep
        # them reachable so it's a drop-in replacement.
        self.assertTrue(hasattr(compile_fn, "cache_info"))
        self.assertEqual(compile_fn.cache_info().misses, 0)

    def test_works_without_cache_info(self):
        # [FUNCTION DOC] A plain callable lacking cache_info is wrapped; the call returns its
        # result, the underlying function ran once with the arguments and one message is logged.
        # A plain callable (no cache_info) must still be timed and reported,
        # just without compiled/cache_hit ground truth (defaults to cache_hit).
        calls = []

        def plain(N, K):
            # [FUNCTION DOC] HELPER: Record the arguments in the calls list and return the string
            # ok.
            calls.append((N, K))
            return "ok"

        compile_fn = instrument_cutedsl_compile("aten::topk")(plain)
        self.assertEqual(compile_fn(256, 64), "ok")
        self.assertEqual(calls, [(256, 64)])
        self.assertEqual(len(self.messages), 1)

    def test_no_work_when_artifact_disabled(self):
        # [FUNCTION DOC] With the artifact disabled (log state preserved and reset), the wrapped
        # function still runs once but no key function call and no log message occur.
        # With the artifact off, the wrapper must run the wrapped fn but skip
        # all instrumentation: no log line, and crucially no key_fn call (user
        # code we shouldn't run for nothing).
        key_calls = []

        def key_fn(N, K):
            # [FUNCTION DOC] HELPER: Record the arguments in key_calls and return the key k.
            key_calls.append((N, K))
            return "k"

        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile("aten::topk", key_fn=key_fn)(fake)

        with preserve_log_state():
            compile_fn(256, 64)

        self.assertEqual(fake.misses, 1)  # wrapped fn still ran
        self.assertEqual(key_calls, [])  # ... but no instrumentation work
        self.assertEqual(self.messages, [])

    def test_compile_event_json_roundtrip(self):
        # [FUNCTION DOC] Builds a compile event with fixed fields and asserts that serializing its
        # dictionary form to JSON and parsing it back preserves op, compiled and misses.
        event = CompileEvent(
            op="aten::topk",
            dsl="cutedsl",
            outcome="compiled",
            compiled=True,
            wall_ms=12.5,
            key="radix N=256 K=64",
            hits=0,
            misses=1,
        )
        loaded = json.loads(json.dumps(event.as_dict(), sort_keys=True))
        self.assertEqual(loaded["op"], "aten::topk")
        self.assertEqual(loaded["compiled"], True)
        self.assertEqual(loaded["misses"], 1)


class TestTritonKernelInstrumentation(_LoggerCaptureTest):
    # [CLASS DOC] Tests for the Triton kernel instrumentation wrapper, run against the fake jit
    # function. GRID is the one-element grid used for launches. Covers compiled and cache_hit
    # reporting, recompiles on new variants, errors, key function use, per-kernel scoping and
    # attribute passthrough.
    GRID = (1,)

    def _kernel(self, op="aten::bmm", key_fn=None):
        # [FUNCTION DOC] HELPER: Return a fake jit function wrapped by the Triton instrumentation
        # for the given op name (default bmm) and optional key function.
        return instrument_triton_kernel(op, key_fn=key_fn)(_FakeJITFunction())

    def test_first_launch_reports_compiled(self):
        # [FUNCTION DOC] First launch returns launched and logs one message with the op, the triton
        # tag and compiled.
        kernel = self._kernel()

        self.assertEqual(kernel[self.GRID](), "launched")

        self.assertEqual(len(self.messages), 1)
        msg = self.messages[0]
        self.assertIn("aten::bmm", msg)
        self.assertIn("[triton]", msg)
        self.assertIn("compiled", msg)

    def test_repeated_launch_reports_cache_hit(self):
        # [FUNCTION DOC] Two launches of the same variant log compiled and then cache_hit.
        kernel = self._kernel()

        kernel[self.GRID](variant="v0")
        kernel[self.GRID](variant="v0")

        self.assertIn("compiled", self.messages[0])
        self.assertIn("cache_hit", self.messages[1])

    def test_new_variant_recompiles(self):
        # [FUNCTION DOC] Launching two different variants logs compiled both times, and the second
        # message shows a running miss count of two.
        kernel = self._kernel()

        kernel[self.GRID](variant="v0")
        kernel[self.GRID](variant="v1")

        for msg in self.messages:
            self.assertIn("compiled", msg)
        # This kernel's running variant count surfaces as misses=2.
        self.assertIn("misses=2", self.messages[1])

    def test_error_is_reported_and_reraised(self):
        # [FUNCTION DOC] A failing launch raises RuntimeError through the wrapper and logs one
        # message containing error.
        fake = _FakeJITFunction()
        fake.raise_on_launch = True
        kernel = instrument_triton_kernel("aten::bmm")(fake)

        with self.assertRaises(RuntimeError):
            kernel[self.GRID]()

        self.assertEqual(len(self.messages), 1)
        self.assertIn("error", self.messages[0])

    def test_key_fn_used_in_log(self):
        # [FUNCTION DOC] A key function receiving the variant keyword makes its text appear in the
        # logged message.
        kernel = self._kernel(key_fn=lambda variant="v0": f"variant={variant}")

        kernel[self.GRID](variant="abc")

        self.assertIn("variant=abc", self.messages[0])

    def test_two_kernels_in_one_module_dont_collide(self):
        # [FUNCTION DOC] Two kernels with separate caches: compiling the second must not make the
        # first appear to recompile. Asserts the first logs compiled then cache_hit, the second
        # logs compiled once, and the first reports its own miss count of one.
        # The core guarantee of per-kernel scoping: each kernel watches only
        # its own cache, so compiling kernel B must NOT make kernel A look
        # like it compiled. The old module-scan summed both and would here
        # falsely report A as "compiled" on its second call.
        kernel_a = self._kernel(op="aten::a", key_fn=lambda variant="v0": "A")
        kernel_b = self._kernel(op="aten::b", key_fn=lambda variant="v0": "B")

        kernel_a[self.GRID](variant="v0")  # A compiles
        kernel_b[self.GRID](variant="v0")  # B compiles (must not touch A)
        kernel_a[self.GRID](variant="v0")  # A hit -- NOT a recompile

        a_msgs = [m for m in self.messages if "aten::a" in m]
        b_msgs = [m for m in self.messages if "aten::b" in m]
        self.assertEqual(len(a_msgs), 2)
        self.assertEqual(len(b_msgs), 1)
        self.assertIn("compiled", a_msgs[0])
        self.assertIn("cache_hit", a_msgs[1])  # would be "compiled" if collided
        self.assertIn("compiled", b_msgs[0])
        # Each reports its own variant count, not the module-wide sum.
        self.assertIn("misses=1", a_msgs[1])

    def test_jit_kernel_exposes_raw_kernel(self):
        # [FUNCTION DOC] Asserts the wrapper exposes the original fake jit function through its
        # jit_kernel attribute.
        # wrap_triton needs the raw JITFunction; the proxy exposes it.
        fake = _FakeJITFunction()
        kernel = instrument_triton_kernel("aten::bmm")(fake)
        self.assertIs(kernel.jit_kernel, fake)

    def test_attribute_passthrough(self):
        # [FUNCTION DOC] Asserts reading an attribute not owned by the wrapper returns the wrapped
        # kernel attribute value.
        # Non-launch attribute access is delegated to the wrapped kernel.
        fake = _FakeJITFunction()
        fake.cache_key = "abc123"
        kernel = instrument_triton_kernel("aten::bmm")(fake)
        self.assertEqual(kernel.cache_key, "abc123")


class TestHelionKernelInstrumentation(_LoggerCaptureTest):
    # [CLASS DOC] Tests for the Helion kernel instrumentation wrapper against the fake Helion
    # kernel: compiled versus cache_hit, recompile on a new variant, error propagation and
    # passthrough of the raw kernel and its attributes.
    def _kernel(self, op="aten::add", key_fn=None):
        # [FUNCTION DOC] HELPER: Return a fake Helion kernel wrapped by the Helion instrumentation
        # for the given op name (default add) and optional key function.
        return instrument_helion_kernel(op, key_fn=key_fn)(_FakeHelionKernel())

    def test_first_call_reports_compiled(self):
        # [FUNCTION DOC] First call returns launched and logs a message with the helion tag,
        # compiled and a miss count of one.
        kernel = self._kernel()
        self.assertEqual(kernel(), "launched")
        self.assertIn("[helion]", self.messages[0])
        self.assertIn("compiled", self.messages[0])
        self.assertIn("misses=1", self.messages[0])

    def test_repeated_call_reports_cache_hit(self):
        # [FUNCTION DOC] Two calls with the same variant log compiled then cache_hit.
        kernel = self._kernel()
        kernel("v0")
        kernel("v0")
        self.assertIn("compiled", self.messages[0])
        self.assertIn("cache_hit", self.messages[1])

    def test_new_variant_recompiles(self):
        # [FUNCTION DOC] Calls with two variants: the second message says compiled and misses two.
        kernel = self._kernel()
        kernel("v0")
        kernel("v1")
        self.assertIn("compiled", self.messages[1])
        self.assertIn("misses=2", self.messages[1])

    def test_error_is_reported_and_reraised(self):
        # [FUNCTION DOC] A failing kernel call re-raises the RuntimeError with its message (helion
        # boom) and the first logged message contains error.
        fake = _FakeHelionKernel()
        fake.raise_on_call = True
        kernel = instrument_helion_kernel("aten::add")(fake)
        with self.assertRaisesRegex(RuntimeError, "helion boom"):
            kernel()
        self.assertIn("error", self.messages[0])

    def test_raw_kernel_and_attributes_are_exposed(self):
        # [FUNCTION DOC] Asserts the wrapper exposes the original fake through helion_kernel and
        # forwards the cache_key attribute.
        fake = _FakeHelionKernel()
        kernel = instrument_helion_kernel("aten::add")(fake)
        self.assertIs(kernel.helion_kernel, fake)
        self.assertEqual(kernel.cache_key, "helion-cache")


class TestTlparseOutput(TestCase):
    # [CLASS DOC] Tests of structured trace output. Intercepts the trace logger: replaces its
    # handlers with a raw file handler (tlparse format), writing to a temporary file, and a capture
    # handler storing records. Tests check emission, metadata and tlparse strict parsing. The setUp
    # and tearDown swap and restore the trace logger state.
    """The instrumentation's whole point on production jobs is tlparse-
    retrievable artifacts. Assert the structured-trace plumbing actually
    fires and that tlparse parses the emitted artifact in --strict mode.
    """

    def setUp(self):
        # [FUNCTION DOC] Enable the artifact, set the trace logger level to debug, save and remove
        # its existing handlers, create a temporary raw file with a stream handler using the trace
        # formatter, and add it and then a capturing handler (order matters because formatting
        # fills in the record metadata). Mutates the global trace logger.
        super().setUp()
        self._log_settings = log_settings(f"+{_ARTIFACT}")
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        # Own trace_log's handler list outright. _init_logs() (run by
        # log_settings) installs LazyTraceHandler, which unregisters itself on
        # its first emit -- and mutating the list mid-dispatch makes
        # Logger.callHandlers skip the handler right after it, silently
        # dropping the first record from ours.
        self._saved_handlers = list(trace_log.handlers)
        for h in self._saved_handlers:
            trace_log.removeHandler(h)

        # Raw trace file in the on-disk format tlparse consumes, written via
        # the same TorchLogsFormatter(trace=True) that TORCH_TRACE installs.
        # NB: this handler must be registered BEFORE the capture handler --
        # TorchLogsFormatter(trace=True) populates record.metadata as a side
        # effect of formatting, and the capture handler reads that field.
        # delete=False: the file is reopened by name (for tlparse and the
        # raw-content read), so don't let close() unlink it; tearDown removes it.
        self.raw_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w", delete=False
        )
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)

        # Capture the records so we can assert on metadata/payload without
        # re-parsing the raw file.
        self.records: list[logging.LogRecord] = []
        self.capture = _CapturingHandler()
        self.capture.records = self.records
        trace_log.addHandler(self.capture)

    def tearDown(self):
        # [FUNCTION DOC] Remove the test handlers, close and delete the temp file, restore the
        # trace logger level and original handlers, close log settings and call the parent.
        trace_log.removeHandler(self.capture)
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        os.unlink(self.raw_file.name)
        trace_log.setLevel(self.old_level)
        for h in self._saved_handlers:
            trace_log.addHandler(h)
        self._log_settings.close()
        super().tearDown()

    def _emit_one(self):
        # [FUNCTION DOC] HELPER: Run one cached CuTeDSL compile through the wrapper (op topk, key
        # function showing N and K) with arguments 256 and 64, then flush the raw file.
        fake = _FakeJitCache()
        compile_fn = instrument_cutedsl_compile(
            "aten::topk", key_fn=lambda N, K: f"radix N={N} K={K}"
        )(fake)
        compile_fn(256, 64)
        self.raw_file.flush()

    def _artifact_records(self):
        # [FUNCTION DOC] HELPER: Return the captured records whose artifact metadata name is
        # native_dsl_compile.
        return [
            r
            for r in self.records
            if getattr(r, "metadata", {}).get("artifact", {}).get("name")
            == "native_dsl_compile"
        ]

    def test_no_artifact_when_disabled(self):
        # [FUNCTION DOC] Regression check: with the artifact disabled, ten emissions produce no
        # artifact records even though the trace logger has handlers.
        # Regression test: a job collecting a structured trace (trace_log has
        # handlers, as it does here) must not get a record per op call without
        # an explicit TORCH_LOGS opt-in.
        with preserve_log_state():
            for _ in range(10):
                self._emit_one()

        self.assertEqual(self._artifact_records(), [])

    def test_emits_artifact_record(self):
        # [FUNCTION DOC] One emission produces one artifact record with json encoding whose payload
        # parses to a mapping with the op, dsl cutedsl, compiled true and the key text.
        self._emit_one()

        recs = self._artifact_records()
        self.assertEqual(len(recs), 1)
        meta = recs[0].metadata["artifact"]
        self.assertEqual(meta["encoding"], "json")
        # Payload must be valid JSON carrying the CompileEvent fields.
        payload = json.loads(recs[0].payload)
        self.assertEqual(payload["op"], "aten::topk")
        self.assertEqual(payload["dsl"], "cutedsl")
        self.assertTrue(payload["compiled"])
        self.assertEqual(payload["key"], "radix N=256 K=64")

    def test_eager_record_has_no_compile_context(self):
        # [FUNCTION DOC] In eager mode (no live compile context) the record metadata has neither a
        # frame id nor a stack entry.
        # In eager dispatch there's no live CompileContext, so the record
        # carries no frame id (and -- via expect_trace_id=False -- no
        # diagnostic stack either).
        self._emit_one()

        meta = self._artifact_records()[0].metadata
        self.assertNotIn("frame_id", meta)
        self.assertNotIn("stack", meta)

    def test_picks_up_live_compile_context(self):
        # [FUNCTION DOC] Within a live compile context with frame id 7 and compile id 3, the record
        # metadata carries those ids and attempt zero.
        # When a native op compiles inside a torch.compile (CompileContext is
        # live), the artifact is auto-tagged with the ambient frame ids so it
        # nests under that compile in tlparse -- like a Dynamo artifact.
        from torch._guards import compile_context, CompileContext, CompileId

        cid = CompileId(frame_id=7, frame_compile_id=3)
        with compile_context(CompileContext(cid)):
            self._emit_one()

        meta = self._artifact_records()[0].metadata
        self.assertEqual(meta["frame_id"], 7)
        self.assertEqual(meta["frame_compile_id"], 3)
        self.assertEqual(meta["attempt"], 0)

    @requires_tlparse
    def test_tlparse_parses_artifact(self):
        # [FUNCTION DOC] Emits one artifact, asserts the raw file contains the artifact name, then
        # runs the tlparse binary in strict mode on it with a temp output directory (removed
        # afterwards); a nonzero exit fails the test. Skipped if tlparse is not installed.
        self._emit_one()

        # Guard against a false pass: --strict over an empty file exits 0, so
        # assert the artifact was actually written before parsing it.
        with open(self.raw_file.name) as f:
            raw = f.read()
        self.assertIn("native_dsl_compile", raw)

        out = tempfile.mkdtemp()
        try:
            # --strict makes tlparse exit non-zero on any unparsable line, so
            # check_call alone is the assertion.
            subprocess.check_call(
                [
                    "tlparse",
                    "-o",
                    out,
                    "--overwrite",
                    "--no-browser",
                    "--strict",
                    self.raw_file.name,
                ]
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)


def _dotted(node):
    # [FUNCTION DOC] HELPER: Return the dotted name of an AST expression. A call is unwrapped to
    # its callee, an attribute joins value and attribute with a dot, a name gives its id; any other
    # node yields an empty string.
    """Dotted name of an AST expression, e.g. ``cute.compile`` / ``triton.jit``.

    Unwraps a Call to its callee, so ``@deco(...)`` and ``deco(...)`` both
    resolve to ``"deco"``. Returns "" for anything we don't model.
    """
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _decorator_names(fn_node):
    # [FUNCTION DOC] HELPER: Return the set of dotted decorator names of a function definition
    # node.
    """Set of dotted decorator names on a FunctionDef node.

    e.g. ``@triton.jit`` -> "triton.jit", ``@instrument_triton_kernel(...)``
    -> "instrument_triton_kernel".
    """
    return {_dotted(d) for d in fn_node.decorator_list}


# One rule per DSL. ``jit_decorator`` is the DSL's raw cache/jit decorator;
# ``instrument_decorator`` is the explicit instrumentation wrapper; and
# ``combined_decorator`` is the one-decorator form that applies both. A
# compile site satisfies the guard if it carries the combined decorator, OR
# both the jit and instrument decorators explicitly. To onboard a new DSL,
# add a row -- every scan and existence check below picks it up. All three
# decorator names must be real entry points in torch._native.instrumentation
# (instrument names) / the DSL (jit name); test_required_decorators_exist
# enforces the instrumentation ones.
_DSL_INSTRUMENTATION_RULES = (
    # (dsl, jit_decorator(s), instrument_decorator, combined_decorator)
    ("triton", "triton.jit", "instrument_triton_kernel", "instrumented_triton_cache"),
    (
        "cutedsl",
        "jit_cache",
        "instrument_cutedsl_compile",
        "instrumented_cutedsl_cache",
    ),
    (
        "helion",
        ("helion.kernel", "helion.jit", "helion.experimental.aot_kernel"),
        "instrument_helion_kernel",
        "instrumented_helion_kernel",
    ),
)


def _is_instrumented(decos, jit_deco, instrument_deco, combined_deco):
    # [FUNCTION DOC] HELPER: Return True if a decorator-name set satisfies a DSL rule: it contains
    # the combined decorator, or it contains one of the jit decorators (a single name or a tuple of
    # names) together with the instrument decorator.
    """True if a function's decorator set satisfies a DSL rule.

    Either the one-shot combined decorator, or the explicit jit + instrument
    stack.
    """
    jit_decos = (jit_deco,) if isinstance(jit_deco, str) else jit_deco
    return combined_deco in decos or (
        any(deco in decos for deco in jit_decos) and instrument_deco in decos
    )


def _scan_for_missing_instrumentation(source, label):
    # [FUNCTION DOC] HELPER: Parse source text and scan every function definition for DSL compile
    # sites, i.e. functions carrying a jit decorator or combined decorator of any rule. Counts the
    # sites and, for each not fully instrumented, adds a message with label, line, function name
    # and the fix to a list. Returns the pair (violation messages, site count). Raises SyntaxError
    # on unparsable source.
    """Return (violations, n_compile_sites) for one Python source string.

    A compile site is any function carrying a DSL rule's ``jit_decorator`` or
    its ``combined_decorator``. A violation is such a site that isn't fully
    instrumented (see :func:`_is_instrumented`). ``n_compile_sites`` lets
    callers assert the scan saw something rather than passing vacuously.
    """
    violations = []
    n_compile_sites = 0
    tree = ast.parse(source, filename=label)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decos = _decorator_names(node)
        for dsl, jit_deco, instrument_deco, combined_deco in _DSL_INSTRUMENTATION_RULES:
            jit_decos = (jit_deco,) if isinstance(jit_deco, str) else jit_deco
            if any(deco in decos for deco in jit_decos) or combined_deco in decos:
                n_compile_sites += 1
                if not _is_instrumented(
                    decos, jit_deco, instrument_deco, combined_deco
                ):
                    violations.append(
                        f"{label}:{node.lineno} {node.name} has an uninstrumented "
                        f"{dsl} decorator but "
                        f"is not instrumented (use @{combined_deco}, or add "
                        f"@{instrument_deco})"
                    )
    return violations, n_compile_sites


# cute.compile must be cached so it is also instrumented: the enclosing
# function must carry @jit_cache or the combined @instrumented_cutedsl_cache.
# A raw call elsewhere would compile uncached *and* invisibly to the
# instrumentation, so it is banned.
_CACHED_COMPILE_CALL = "cute.compile"
_CACHE_DECORATORS = ("jit_cache", "instrumented_cutedsl_cache")


def _scan_for_raw_cute_compile(source, label):
    # [FUNCTION DOC] HELPER: Parse source and find calls to the cute compile function. Builds a
    # child to parent map of the tree, and for each call finds its nearest enclosing function; a
    # call is a violation if that function lacks a cache decorator or the call is at module level.
    # Returns the pair (violation messages, number of compile calls seen).
    """Return (violations, n_calls) for one Python source string.

    A violation is a ``cute.compile`` call whose nearest enclosing function
    carries neither ``@jit_cache`` nor ``@instrumented_cutedsl_cache`` (or
    that sits at module level). ``n_calls`` counts compile calls seen.
    """
    tree = ast.parse(source, filename=label)
    # Map each node to its parent so we can climb to the nearest FunctionDef.
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def enclosing_function(node):
        # [FUNCTION DOC] HELPER: Climb the parent map from a node and return the nearest enclosing
        # function definition, or None if there is none.
        cur = parent.get(node)
        while cur is not None:
            if isinstance(cur, ast.FunctionDef):
                return cur
            cur = parent.get(cur)
        return None

    violations = []
    n_calls = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _dotted(node) == _CACHED_COMPILE_CALL):
            continue
        n_calls += 1
        fn = enclosing_function(node)
        cached = fn is not None and any(
            d in _decorator_names(fn) for d in _CACHE_DECORATORS
        )
        if not cached:
            where = fn.name if fn is not None else "<module level>"
            allowed = " or ".join(f"@{d}" for d in _CACHE_DECORATORS)
            violations.append(
                f"{label}:{node.lineno} {_CACHED_COMPILE_CALL}() in {where} is "
                f"not wrapped by {allowed}"
            )
    return violations, n_calls


class TestInstrumentationCoverage(TestCase):
    # [CLASS DOC] CI guard test case: static AST scan of the torch/_native/ops sources ensuring
    # each DSL compile site is instrumented and each cute compile call is cached, plus meta-tests
    # proving the scanners flag bad inputs. Stateless apart from a file generator.
    """CI guard: every DSL compile site must carry its instrumentation.

    Static AST scan of ``torch/_native/ops`` -- adding a kernel for a known
    DSL (per :data:`_DSL_INSTRUMENTATION_RULES`) without the matching
    instrumentation decorator turns this test (and thus CI) red.

    Extending to a new DSL: add a row to ``_DSL_INSTRUMENTATION_RULES``.

    For CuTeDSL this also enforces the full chain: every ``cute.compile()``
    must be wrapped by ``@jit_cache`` (``test_no_raw_cute_compile_calls``),
    and every ``@jit_cache`` must be instrumented -- so no compile can escape
    caching or instrumentation.

    Out of scope by construction: ops that wrap *vendored* compile fns at the
    call site (e.g. norm's rmsnorm, which has no local ``@jit_cache`` or
    ``cute.compile``) aren't decorator/call sites here -- they have their own
    tests.
    """

    def _ops_files(self):
        # [FUNCTION DOC] HELPER: Generator yielding the path of every Python file under the ops
        # directory of the torch._native package, found by walking it recursively.
        import torch._native

        ops_dir = os.path.join(os.path.dirname(torch._native.__file__), "ops")
        for root, _, files in os.walk(ops_dir):
            for name in files:
                if name.endswith(".py"):
                    yield os.path.join(root, name)

    def test_required_decorators_exist(self):
        # [FUNCTION DOC] Asserts that every instrument and combined decorator named by the rule
        # table exists as an attribute of the instrumentation module.
        # Ties each rule to the real API: a typo'd or removed instrumentation
        # entry point fails here rather than letting the scan pass vacuously.
        import torch._native.instrumentation as instr

        for dsl, _, instrument_deco, combined_deco in _DSL_INSTRUMENTATION_RULES:
            for name in (instrument_deco, combined_deco):
                self.assertTrue(
                    hasattr(instr, name),
                    f"rule for DSL {dsl!r} names {name!r}, which is not in "
                    f"torch._native.instrumentation",
                )

    def test_every_compile_site_is_instrumented(self):
        # [FUNCTION DOC] Scans all ops files with the instrumentation scanner, collecting
        # violations and the number of sites; asserts at least one site was seen and no violations.
        missing = []
        checked = 0
        for path in self._ops_files():
            with open(path) as f:
                violations, n = _scan_for_missing_instrumentation(
                    f.read(), os.path.relpath(path)
                )
            missing += violations
            checked += n

        self.assertTrue(checked, "scan found no compile sites -- test is stale")
        self.assertEqual(
            missing,
            [],
            "DSL compile sites missing instrumentation:\n" + "\n".join(missing),
        )

    def test_scan_flags_uninstrumented_kernel_per_dsl(self):
        # [FUNCTION DOC] Meta-test with source templates per DSL (triton, cutedsl, helion). Asserts
        # a bare jit decorator is counted and flagged, while explicit jit-plus-instrument and
        # combined decorator forms are counted and not flagged. Also checks the Helion AOT kernel
        # and plain Helion jit decorators are flagged.
        # Meta-test: prove the guard fires. For each DSL: a bare jit decorator
        # (no instrumentation) must be flagged, while BOTH accepted forms --
        # the explicit jit+instrument stack and the one-shot combined
        # decorator -- must pass.
        templates = {
            # dsl: (bad, explicit_ok, combined_ok)
            "triton": (
                "@triton.jit\ndef k(): ...\n",
                "@instrument_triton_kernel('aten::x')\n@triton.jit\ndef k(): ...\n",
                "@instrumented_triton_cache('aten::x')\ndef k(): ...\n",
            ),
            "cutedsl": (
                "@jit_cache\ndef c(): ...\n",
                "@instrument_cutedsl_compile('aten::x')\n@jit_cache\ndef c(): ...\n",
                "@instrumented_cutedsl_cache('aten::x')\ndef c(): ...\n",
            ),
            "helion": (
                "@helion.kernel\ndef h(): ...\n",
                "@instrument_helion_kernel('aten::x')\n@helion.kernel\ndef h(): ...\n",
                "@instrumented_helion_kernel('aten::x')\ndef h(): ...\n",
            ),
        }
        for dsl, (bad, explicit_ok, combined_ok) in templates.items():
            bad_v, bad_n = _scan_for_missing_instrumentation(bad, f"<{dsl}-bad>")
            self.assertEqual(bad_n, 1, f"{dsl}: scan didn't see the compile site")
            self.assertEqual(
                len(bad_v), 1, f"{dsl}: uninstrumented kernel was NOT flagged"
            )

            for form, src in (("explicit", explicit_ok), ("combined", combined_ok)):
                v, n = _scan_for_missing_instrumentation(src, f"<{dsl}-{form}>")
                self.assertEqual(n, 1, f"{dsl}/{form}: scan missed the site")
                self.assertEqual(v, [], f"{dsl}/{form}: wrongly flagged")

        aot_v, aot_n = _scan_for_missing_instrumentation(
            "@helion.experimental.aot_kernel\ndef h(): ...\n", "<helion-aot-bad>"
        )
        self.assertEqual(aot_n, 1)
        self.assertEqual(len(aot_v), 1)

        jit_v, jit_n = _scan_for_missing_instrumentation(
            "@helion.jit\ndef h(): ...\n", "<helion-jit-bad>"
        )
        self.assertEqual(jit_n, 1)
        self.assertEqual(len(jit_v), 1)

    def test_no_raw_cute_compile_calls(self):
        # [FUNCTION DOC] Scans all ops files for cute compile calls outside a cache-decorated
        # function; asserts calls were seen and no violation exists.
        # Caching is compulsory for cutedsl compiles: every cute.compile() must
        # sit inside a function decorated with @jit_cache or the combined
        # @instrumented_cutedsl_cache. A raw call would be uncached and
        # invisible to instrumentation.
        bad = []
        seen = 0
        for path in self._ops_files():
            with open(path) as f:
                violations, n = _scan_for_raw_cute_compile(
                    f.read(), os.path.relpath(path)
                )
            bad += violations
            seen += n

        self.assertTrue(seen, "scan found no cute.compile calls -- test is stale")
        self.assertEqual(
            bad,
            [],
            "uncached cute.compile() calls:\n" + "\n".join(bad),
        )

    def test_scan_flags_raw_cute_compile(self):
        # [FUNCTION DOC] Meta-test for the raw compile scanner: a call in a plain function and a
        # module-level call are each counted and flagged, while a call in a function decorated with
        # the jit cache or the combined decorator is counted and not flagged.
        # Meta-test: prove the caching requirement fires, and that both the raw
        # @jit_cache and the combined decorator satisfy it.
        raw = "def f():\n    return cute.compile(k)\n"
        raw_v, raw_n = _scan_for_raw_cute_compile(raw, "<raw>")
        self.assertEqual(raw_n, 1, "scan didn't see the cute.compile call")
        self.assertEqual(len(raw_v), 1, "raw cute.compile was NOT flagged")

        module_level = "x = cute.compile(k)\n"
        ml_v, ml_n = _scan_for_raw_cute_compile(module_level, "<module>")
        self.assertEqual(ml_n, 1)
        self.assertEqual(len(ml_v), 1, "module-level cute.compile was NOT flagged")

        for form, deco in (
            ("jit_cache", "@jit_cache"),
            ("combined", "@instrumented_cutedsl_cache('aten::x')"),
        ):
            src = f"{deco}\ndef f():\n    return cute.compile(k)\n"
            v, n = _scan_for_raw_cute_compile(src, f"<{form}>")
            self.assertEqual(n, 1)
            self.assertEqual(v, [], f"{form}-wrapped cute.compile was wrongly flagged")


if __name__ == "__main__":
    run_tests()
