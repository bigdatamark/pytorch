# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for torch._inductor.runtime.benchmarking: the Benchmarker, TritonBenchmarker,
# InductorBenchmarker and TorchProfilerBenchmarker classes, the per-device dispatch registry, the
# optional GPU benchmark lock context, profiler kernel-time extraction, and CUDA graph
# benchmarking. Most tests use fakes and patches so they run without a GPU. Defines
# ALL_BENCHMARKER_CLASSES (the base and Triton benchmarkers) used for parametrization. No import-
# time side effects beyond imports.

import contextlib
import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._inductor.config import (
    inductor_default_autotune_rep,
    inductor_default_autotune_warmup,
)
from torch._inductor.runtime.benchmarking import (
    Benchmarker,
    InductorBenchmarker,
    TorchProfilerBenchmarker,
    TritonBenchmarker,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


ALL_BENCHMARKER_CLASSES = (
    Benchmarker,
    TritonBenchmarker,
)


@instantiate_parametrized_tests
class TestBenchmarker(TestCase):
    # [CLASS DOC] Parametrized test case for the benchmarker classes. setUp seeds the random
    # generator and clears dynamo counters; tests read per-benchmarker call counters from the
    # inductor counters.
    def setUp(self):
        # [FUNCTION DOC] Calls base setUp, seeds torch with 12345 and clears the dynamo counters.
        super().setUp()
        torch.manual_seed(12345)
        counters.clear()

    @staticmethod
    def get_counter_value(benchmarker_cls, fn_name):
        # [FUNCTION DOC] HELPER: Returns the inductor counter named benchmarking.<class
        # name>.<function name> for the given benchmarker class and function name.
        return counters["inductor"][
            f"benchmarking.{benchmarker_cls.__name__}.{fn_name}"
        ]

    @staticmethod
    def make_params(device, size=100):
        # [FUNCTION DOC] HELPER: Builds a sum of a random tensor of the given size on the device.
        # Returns a pair: the triple of function, positional args tuple and empty kwargs dict, and
        # a zero-argument callable that runs the function with those arguments.
        fn, fn_args, fn_kwargs = torch.sum, (torch.randn(size, device=device),), {}
        _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
        return (fn, fn_args, fn_kwargs), _callable

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["benchmarker_cls"] is Benchmarker
        and params["device"] == GPU_TYPE,
    )
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_benchmark_smoke(self, benchmarker_cls, device):
        # [FUNCTION DOC] Needs CPU and GPU. Runs benchmark on a sum for the benchmarker class and
        # device, checks the timing is positive, the benchmark counter is 1, and the benchmark_cpu
        # or benchmark_gpu counter (by device) is 1. Marked as an expected failure for the base
        # Benchmarker on the GPU device.
        benchmarker = benchmarker_cls()
        (fn, fn_args, fn_kwargs), _ = self.make_params(device)
        timing = benchmarker.benchmark(fn, fn_args, fn_kwargs)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark"), 1)
        self.assertEqual(
            self.get_counter_value(
                benchmarker_cls, "benchmark_cpu" if device == "cpu" else "benchmark_gpu"
            ),
            1,
        )

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_cpu_smoke(self, benchmarker_cls, device="cpu"):
        # [FUNCTION DOC] Needs CPU. Calls benchmark_cpu directly on a CPU sum callable; checks the
        # timing is positive and the benchmark_cpu counter is 1.
        benchmarker = benchmarker_cls()
        _, _callable = self.make_params(device)
        timing = benchmarker.benchmark_cpu(_callable)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark_cpu"), 1)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["benchmarker_cls"] is Benchmarker,
    )
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_gpu_smoke(self, benchmarker_cls, device=GPU_TYPE):
        # [FUNCTION DOC] Needs GPU. Calls benchmark_gpu directly on a GPU sum callable; checks a
        # positive timing and benchmark_gpu counter of 1. Expected failure for the base
        # Benchmarker.
        benchmarker = benchmarker_cls()
        _, _callable = self.make_params(device)
        timing = benchmarker.benchmark_gpu(_callable)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark_gpu"), 1)

    @unittest.skipIf(not HAS_CPU and not HAS_GPU, "requires CPU or GPU")
    @unittest.expectedFailure
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_safely_infers_device_no_devices(
        self, benchmarker_cls, device="cpu" if HAS_CPU else GPU_TYPE
    ):
        # [FUNCTION DOC] Expected-failure test: calls benchmark with a function but no tensor
        # arguments, so no device can be inferred from the arguments, and the call is expected to
        # raise.
        benchmarker = benchmarker_cls()
        (fn, _, _), _ = self.make_params(device)
        benchmarker.benchmark(fn, (), {})

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @unittest.expectedFailure
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_safely_infers_device_many_devices(self, benchmarker_cls):
        # [FUNCTION DOC] Expected-failure test: merges CPU and GPU tensor arguments into one call
        # so the device is ambiguous, and benchmark is expected to raise. KNOWN ISSUE: it calls
        # make_sum, which does not exist in this class (make_params is the helper), so it fails
        # with AttributeError and the expected failure marker hides that the intended condition is
        # never exercised.
        benchmarker = benchmarker_cls()
        (fn, cpu_args, cpu_kwargs), _ = self.make_sum("cpu")
        (_, gpu_args, gpu_kwargs), _ = self.make_sum(GPU_TYPE)
        many_devices_args = cpu_args + gpu_args
        many_devices_kwargs = cpu_kwargs
        many_devices_kwargs.update(gpu_kwargs)
        benchmarker.benchmark(fn, many_devices_args, many_devices_kwargs)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_benchmark_warmup_and_rep_defaults(self):
        # [FUNCTION DOC] Needs GPU. Replaces TritonBenchmarker.benchmark_gpu with a recorder and
        # checks that benchmark, called without warmup or rep, forwards the config defaults
        # inductor_default_autotune_warmup and inductor_default_autotune_rep to it.
        """Test that benchmark_gpu receives default warmup and rep values when not specified."""
        captured_kwargs = {}

        def capture_benchmark_gpu(self, _callable, **kwargs):
            # [FUNCTION DOC] HELPER: Fake benchmark_gpu that stores the keyword arguments in the
            # enclosing dict and returns the dummy timing 1.0.
            captured_kwargs.update(kwargs)
            return 1.0  # Return a dummy timing

        benchmarker = TritonBenchmarker()
        (fn, fn_args, fn_kwargs), _ = self.make_params(GPU_TYPE)

        with patch.object(TritonBenchmarker, "benchmark_gpu", capture_benchmark_gpu):
            benchmarker.benchmark(fn, fn_args, fn_kwargs)

        self.assertEqual(captured_kwargs["warmup"], inductor_default_autotune_warmup)
        self.assertEqual(captured_kwargs["rep"], inductor_default_autotune_rep)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_benchmark_warmup_and_rep_custom_values(self):
        # [FUNCTION DOC] Needs GPU. Like the defaults test but passes warmup 50 and rep 200
        # explicitly and checks those values reach benchmark_gpu unchanged.
        """Test that benchmark_gpu receives custom warmup and rep values when specified."""
        captured_kwargs = {}

        def capture_benchmark_gpu(self, _callable, **kwargs):
            # [FUNCTION DOC] HELPER: Fake benchmark_gpu that stores the keyword arguments in the
            # enclosing dict and returns the dummy timing 1.0.
            captured_kwargs.update(kwargs)
            return 1.0  # Return a dummy timing

        benchmarker = TritonBenchmarker()
        (fn, fn_args, fn_kwargs), _ = self.make_params(GPU_TYPE)

        custom_warmup = 50
        custom_rep = 200

        with patch.object(TritonBenchmarker, "benchmark_gpu", capture_benchmark_gpu):
            benchmarker.benchmark(
                fn, fn_args, fn_kwargs, warmup=custom_warmup, rep=custom_rep
            )

        self.assertEqual(captured_kwargs["warmup"], custom_warmup)
        self.assertEqual(captured_kwargs["rep"], custom_rep)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmarker_cpu_override_dispatch(self, benchmarker_cls, device="cpu"):
        # [FUNCTION DOC] Needs CPU. Registers a custom handler for the cpu device with override,
        # with the real benchmark_cpu and benchmark_gpu patched to raise, and checks benchmark
        # returns the handler's result exactly once. Snapshots the global dispatch registry first
        # and restores it in a finally block.
        # Registers a custom handler for 'cpu' and verifies dispatch uses it instead of the default path.
        from torch._inductor.runtime import benchmarking as _bench

        benchmarker = benchmarker_cls()

        # Snapshot registry and restore at the end to avoid cross-test pollution.
        orig = dict(_bench._BENCHMARK_DISPATCH)
        try:
            seen = {"cpu_override": 0}

            def custom_cpu(self, fn, *, warmup, rep, **kw):
                # [FUNCTION DOC] HELPER: Fake cpu handler: counts its call in the enclosing dict
                # and returns a marker string.
                seen["cpu_override"] += 1
                return "cpu-override"

            # Override the built-in 'cpu' registration
            _bench.register_benchmarker("cpu", custom_cpu, override=True)

            # Ensure default CPU/GPU methods are NOT called if registry override works.
            with (
                patch.object(
                    benchmarker_cls,
                    "benchmark_cpu",
                    side_effect=AssertionError(
                        "benchmark_cpu should not be called when a custom 'cpu' handler is registered"
                    ),
                    create=True,
                ),
                patch.object(
                    benchmarker_cls,
                    "benchmark_gpu",
                    side_effect=AssertionError(
                        "benchmark_gpu should not be called for 'cpu' device"
                    ),
                    create=True,
                ),
            ):
                (fn, fn_args, fn_kwargs), _ = self.make_params(device)
                out = benchmarker.benchmark(fn, fn_args, fn_kwargs)
                self.assertEqual(out, "cpu-override")
                self.assertEqual(seen["cpu_override"], 1)
        finally:
            _bench._BENCHMARK_DISPATCH.clear()
            _bench._BENCHMARK_DISPATCH.update(orig)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmarker_cpu_override_runs_callable(
        self, benchmarker_cls, device="cpu"
    ):
        # [FUNCTION DOC] Needs CPU. Overrides the cpu handler with one that delegates to the
        # benchmarker's benchmark_cpu, then checks benchmark with warmup 1 and rep 1 returns a
        # positive time (the user function is actually run and no TypeError occurs). The dispatch
        # registry is restored afterwards.
        from torch._inductor.runtime import benchmarking as _bench

        benchmarker = benchmarker_cls()
        orig = dict(_bench._BENCHMARK_DISPATCH)
        try:
            # Override CPU but still route to benchmark_cpu internally
            def custom_cpu(self, f, *, warmup, rep, **kw):
                # [FUNCTION DOC] HELPER: Fake cpu handler that forwards the callable and settings
                # to benchmark_cpu.
                # Just delegate to the original path; we want to ensure `f()` calls the user's fn.
                return self.benchmark_cpu(f, warmup=warmup, rep=rep, **kw)

            _bench.register_benchmarker("cpu", custom_cpu, override=True)
            # Define a simple op and ensure it actually runs without TypeError
            (fn, fn_args, fn_kwargs), _ = self.make_params(device)
            out = benchmarker.benchmark(fn, fn_args, fn_kwargs, warmup=1, rep=1)
            self.assertGreater(out, 0)
        finally:
            _bench._BENCHMARK_DISPATCH.clear()
            _bench._BENCHMARK_DISPATCH.update(orig)

    def test_default_profiler_benchmarker_selection_supports_xpu_only(self):
        # [FUNCTION DOC] With the profiler benchmarker config on and the experimental one off,
        # checks the default benchmarker factory returns a TorchProfilerBenchmarker without
        # querying CUDA, XPU or MTIA availability (those availability checks are patched to
        # raise).
        from torch._inductor.runtime import benchmarking as _bench

        with (
            patch.object(
                _bench.inductor_config, "use_torch_profiler_benchmarker", True
            ),
            patch.object(_bench.inductor_config, "use_experimental_benchmarker", False),
            patch.object(
                _bench.torch.cuda,
                "is_available",
                side_effect=AssertionError("should not query CUDA availability"),
            ),
            patch.object(
                _bench.torch.xpu,
                "is_available",
                side_effect=AssertionError("should not query XPU availability"),
            ),
            patch.object(
                _bench.torch.mtia,
                "is_available",
                side_effect=AssertionError("should not query MTIA availability"),
            ),
        ):
            self.assertIsInstance(
                _bench._make_default_benchmarker(), TorchProfilerBenchmarker
            )

    def test_profiler_benchmarker_fallback_uses_correlated_device_events(self):
        # [FUNCTION DOC] Builds fake profiler and Kineto events (a CPU callable event with one
        # child, plus XPU events: an unrelated one, a kernel linked to the child, and a user-
        # annotation event) and checks the device kernel time helper returns 5.0 microseconds,
        # i.e. only the kernel linked to the callable.
        from torch._inductor.runtime import benchmarking as _bench
        from torch.autograd import DeviceType

        class FakeProfilerEvent:
            # [CLASS DOC] HELPER: Fake profiler event with name, device_type, id and a list of
            # cpu_children.
            def __init__(self, name, device_type, event_id, cpu_children=()):
                # [FUNCTION DOC] HELPER: Stores the given fields; children are copied into a list.
                self.name = name
                self.device_type = device_type
                self.id = event_id
                self.cpu_children = list(cpu_children)

        class FakeKinetoEvent:
            # [CLASS DOC] HELPER: Fake Kineto event exposing stored values through accessor
            # methods (name, device type, linked and own correlation ids, activity type, start and
            # end in nanoseconds).
            def __init__(
                self,
                name,
                device_type,
                linked_correlation_id,
                correlation_id,
                activity_type,
                start_ns,
                end_ns,
            ):
                # [FUNCTION DOC] HELPER: Stores the given fields in underscore-prefixed
                # attributes.
                self._name = name
                self._device_type = device_type
                self._linked_correlation_id = linked_correlation_id
                self._correlation_id = correlation_id
                self._activity_type = activity_type
                self._start_ns = start_ns
                self._end_ns = end_ns

            def name(self):
                # [FUNCTION DOC] HELPER: Returns the stored event name.
                return self._name

            def device_type(self):
                # [FUNCTION DOC] HELPER: Returns the stored device type.
                return self._device_type

            def linked_correlation_id(self):
                # [FUNCTION DOC] HELPER: Returns the stored linked correlation id.
                return self._linked_correlation_id

            def correlation_id(self):
                # [FUNCTION DOC] HELPER: Returns the stored correlation id.
                return self._correlation_id

            def activity_type(self):
                # [FUNCTION DOC] HELPER: Returns the stored activity type.
                return self._activity_type

            def start_ns(self):
                # [FUNCTION DOC] HELPER: Returns the stored start time in nanoseconds.
                return self._start_ns

            def end_ns(self):
                # [FUNCTION DOC] HELPER: Returns the stored end time in nanoseconds.
                return self._end_ns

        child_event = FakeProfilerEvent("aten::sum", DeviceType.CPU, 11)
        profiler_events = [
            FakeProfilerEvent(
                _bench._CALLABLE_PROFILE_EVENT_NAME,
                DeviceType.CPU,
                7,
                (child_event,),
            )
        ]
        kineto_events = [
            FakeKinetoEvent("unrelated", DeviceType.XPU, 0, 99, "kernel", 0, 1000),
            FakeKinetoEvent(
                "xpu_kernel", DeviceType.XPU, 11, 101, "kernel", 1000, 6000
            ),
            FakeKinetoEvent(
                _bench._CALLABLE_PROFILE_EVENT_NAME,
                DeviceType.XPU,
                7,
                7,
                "gpu_user_annotation",
                1000,
                9000,
            ),
        ]

        self.assertEqual(
            _bench._get_callable_device_kernel_time_us(
                kineto_events, profiler_events, 1, DeviceType.XPU
            ),
            5.0,
        )

    def test_gpu_benchmark_lock_without_registered_context_is_noop(self):
        # [FUNCTION DOC] With no lock context registered, entering the maybe-lock manager does
        # nothing and does not query the current CUDA device (that call is patched to raise).
        from torch._inductor.runtime import benchmarking as _bench

        with patch(
            "torch.cuda.current_device",
            side_effect=AssertionError("should not query device"),
        ):
            with _bench.maybe_gpu_benchmark_lock():
                pass

    def test_gpu_benchmark_lock_uses_registered_context(self):
        # [FUNCTION DOC] Registers a recording context manager as the lock context and checks the
        # body of the maybe-lock manager runs between its enter and exit (recorded order: enter,
        # body, exit) without querying the CUDA device. The previous context is restored in a
        # finally block.
        from torch._inductor.runtime import benchmarking as _bench

        calls = []

        @contextlib.contextmanager
        def custom_context():
            # [FUNCTION DOC] HELPER: Context manager that records enter and exit in the enclosing
            # list around its yield.
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

        previous = _bench.set_gpu_benchmark_lock_context(custom_context)
        try:
            with patch(
                "torch.cuda.current_device",
                side_effect=AssertionError("custom context should not query device"),
            ):
                with _bench.maybe_gpu_benchmark_lock():
                    calls.append("body")
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(calls, ["enter", "body", "exit"])

    def test_set_gpu_benchmark_lock_context_returns_previous_context(self):
        # [FUNCTION DOC] Checks that setting a new lock context returns the previously registered
        # one, and restores the original afterwards.
        from torch._inductor.runtime import benchmarking as _bench

        @contextlib.contextmanager
        def first_context():
            # [FUNCTION DOC] HELPER: Context manager that does nothing.
            yield

        @contextlib.contextmanager
        def second_context():
            # [FUNCTION DOC] HELPER: Context manager that does nothing.
            yield

        previous = _bench.set_gpu_benchmark_lock_context(first_context)
        try:
            self.assertIs(
                _bench.set_gpu_benchmark_lock_context(second_context),
                first_context,
            )
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

    def test_do_bench_using_profiling_uses_gpu_benchmark_lock(self):
        # [FUNCTION DOC] Registers a recording lock context, patches the profiling-based timing
        # helper in inductor utils with a fake that records its warmup, rep and vetted flag and
        # runs the callable, then checks do_bench_using_profiling returns 3.0 and that the order
        # is enter, fake call with warmup 1, rep 2 and vetted true, the callable, exit, i.e. the
        # lock wraps the timing call.
        from torch._inductor import utils as inductor_utils
        from torch._inductor.runtime import benchmarking as _bench

        calls = []

        @contextlib.contextmanager
        def custom_context():
            # [FUNCTION DOC] HELPER: Context manager that records enter and exit in the enclosing
            # list around its yield.
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

        def fake_do_bench(fn, warmup, rep, is_vetted_benchmarking):
            # [FUNCTION DOC] HELPER: Fake profiling timer: records its warmup, rep and vetted
            # flag, calls the benchmarked function once and returns 3.0.
            calls.append((warmup, rep, is_vetted_benchmarking))
            fn()
            return 3.0

        previous = _bench.set_gpu_benchmark_lock_context(custom_context)
        try:
            with patch.object(
                inductor_utils,
                "_do_bench_using_profiling",
                side_effect=fake_do_bench,
            ):
                result = inductor_utils.do_bench_using_profiling(
                    lambda: calls.append("fn"),
                    warmup=1,
                    rep=2,
                    is_vetted_benchmarking=True,
                )
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(result, 3.0)
        self.assertEqual(calls, ["enter", (1, 2, True), "fn", "exit"])

    def test_benchmark_gpu_with_cuda_graph_uses_gpu_benchmark_lock(self):
        # [FUNCTION DOC] With CUDA APIs patched to fakes and a recording lock context, runs the
        # CUDA graph benchmark on a fake benchmarker whose benchmark_gpu is lock-decorated, and
        # checks the result 9.0 and the recorded sequence: lock entered, three direct calls
        # (warmup, side stream, capture), lock entered again, benchmark_gpu, graph replay, then
        # both locks exited (re-entrant nesting).
        from torch._inductor.runtime import benchmarking as _bench

        class FakeCUDAGraph:
            # [CLASS DOC] HELPER: Fake CUDA graph whose replay records a replay entry in the
            # enclosing list.
            def replay(self):
                # [FUNCTION DOC] HELPER: Appends a replay marker to the enclosing calls list.
                calls.append("replay")

        class FakeBenchmarker(Benchmarker):
            # [CLASS DOC] HELPER: Benchmarker subclass whose lock-decorated benchmark_gpu records
            # itself and returns 9.0.
            @_bench.gpu_benchmark_lock
            def benchmark_gpu(self, _callable, **kwargs):
                # [FUNCTION DOC] HELPER: Records a benchmark_gpu marker, runs the callable once
                # and returns 9.0.
                calls.append("benchmark_gpu")
                _callable()
                return 9.0

        benchmarker = FakeBenchmarker()
        calls = []

        @contextlib.contextmanager
        def custom_context():
            # [FUNCTION DOC] HELPER: Context manager that records enter and exit in the enclosing
            # list around its yield.
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

        class FakeStream:
            # [CLASS DOC] HELPER: Fake CUDA stream with no-op wait_stream and synchronize methods.
            def wait_stream(self, stream):
                # [FUNCTION DOC] HELPER: No-op replacement for wait_stream.
                pass

            def synchronize(self):
                # [FUNCTION DOC] HELPER: No-op replacement for synchronize.
                pass

        current_stream = FakeStream()

        previous = _bench.set_gpu_benchmark_lock_context(custom_context)
        try:
            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.Stream", FakeStream),
                patch("torch.cuda.current_stream", return_value=current_stream),
                patch("torch.cuda.stream", return_value=contextlib.nullcontext()),
                patch("torch.cuda.CUDAGraph", FakeCUDAGraph),
                patch("torch.cuda.graph", return_value=contextlib.nullcontext()),
            ):
                result = benchmarker.benchmark_gpu_with_cuda_graph(
                    lambda: calls.append("call")
                )
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(result, 9.0)
        self.assertEqual(
            calls,
            [
                "enter",
                "call",
                "call",
                "call",
                "enter",
                "benchmark_gpu",
                "replay",
                "exit",
                "exit",
            ],
        )

    def test_autotune_cudagraph_benchmarking_requires_max_autotune(self):
        # [FUNCTION DOC] Runs InductorBenchmarker.benchmark_gpu (via a fake subclass with canned
        # cache size, events and graph benchmark) with cudagraph benchmarking on. With
        # max_autotune off the CUDA graph path must not be taken: result 1.0 from the event
        # timings and the callable is called three times. With max_autotune on it must be taken:
        # result 7.0, one graph benchmark call, callable not invoked.
        from torch._inductor.runtime import benchmarking as _bench

        class FakeBuffer:
            def zero_(self):
                # [CLASS DOC] HELPER: Fake buffer with a no-op zero_ method.
                pass
                # [FUNCTION DOC] HELPER: No-op replacement for zero_.

        class FakeDeviceInterface:
            # [CLASS DOC] HELPER: Fake device interface with a no-op synchronize method.
            def synchronize(self):
                # [FUNCTION DOC] HELPER: No-op replacement for synchronize.
                pass

        class FakeEvent:
            # [CLASS DOC] HELPER: Fake timing event whose elapsed time is always 1.0.
            def record(self):
                # [FUNCTION DOC] HELPER: No-op replacement for record.
                pass

            def elapsed_time(self, other):
                # [FUNCTION DOC] HELPER: Returns 1.0 regardless of the other event.
                return 1.0

        class FakeInductorBenchmarker(InductorBenchmarker):
            # [CLASS DOC] HELPER: InductorBenchmarker subclass with canned device cache size (4),
            # fake event pairs, and a graph benchmark stub; counts graph benchmark calls in an
            # attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Runs the parent constructor and sets the graph call
                # counter to zero.
                super().__init__()
                self.cuda_graph_benchmark_calls = 0

            def benchmark_gpu_with_cuda_graph(self, _callable, **kwargs):
                # [FUNCTION DOC] HELPER: Increments the graph call counter and returns 7.0.
                self.cuda_graph_benchmark_calls += 1
                return 7.0

            def get_device_cache_size(self, device_type=None):
                # [FUNCTION DOC] HELPER: Returns the fixed cache size 4.
                return 4

            def get_event_pairs(self, iters, device_type=None):
                # [FUNCTION DOC] HELPER: Returns the requested number of pairs of fake events.
                return [(FakeEvent(), FakeEvent()) for _ in range(iters)]

        def run_benchmark(max_autotune):
            # [FUNCTION DOC] HELPER: Runs benchmark_gpu of a fresh fake benchmarker for the cuda
            # device type with the device interface, buffer allocation and the two config flags
            # patched (cudagraph benchmarking on, max_autotune as given), using one iteration of
            # each phase. Returns the result, the graph benchmark call count and the list of
            # callable invocations.
            benchmarker = FakeInductorBenchmarker()
            callable_calls = []
            with (
                patch.object(
                    _bench,
                    "get_interface_for_device",
                    return_value=FakeDeviceInterface(),
                ),
                patch.object(_bench.torch, "empty", return_value=FakeBuffer()),
                patch.object(
                    _bench.inductor_config, "autotune_cudagraph_benchmarking", True
                ),
                patch.object(_bench.inductor_config, "max_autotune", max_autotune),
            ):
                result = benchmarker.benchmark_gpu(
                    lambda: callable_calls.append("call"),
                    estimation_iters=1,
                    memory_warmup_iters=1,
                    benchmark_iters=1,
                    is_vetted_benchmarking=True,
                    device_type="cuda",
                )
            return result, benchmarker.cuda_graph_benchmark_calls, callable_calls

        result, cuda_graph_calls, callable_calls = run_benchmark(max_autotune=False)
        self.assertEqual(result, 1.0)
        self.assertEqual(cuda_graph_calls, 0)
        self.assertEqual(callable_calls, ["call", "call", "call"])

        result, cuda_graph_calls, callable_calls = run_benchmark(max_autotune=True)
        self.assertEqual(result, 7.0)
        self.assertEqual(cuda_graph_calls, 1)
        self.assertEqual(callable_calls, [])

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @parametrize(
        "hip_value, expected_buffer_size_bytes",
        ((None, 1024), ("mock-hip", 256 * 1024 * 1024)),
    )
    def test_torch_profiler_benchmarker_reuses_inductor_helpers(
        self, hip_value, expected_buffer_size_bytes, device=GPU_TYPE
    ):
        # [FUNCTION DOC] Needs GPU. Parametrized on a mocked HIP version. Sets a small cached L2
        # cache size on a TorchProfilerBenchmarker, runs benchmark_gpu with torch.empty spied, and
        # checks the timing is positive, get_event_pairs was called once with 1 and the device
        # type, and the first flush buffer has expected bytes divided by four elements: 1024 when
        # HIP is unset, 256 MB when HIP is set.
        benchmarker = TorchProfilerBenchmarker()
        benchmarker.__dict__["L2_cache_size"] = 1024
        _, _callable = self.make_params(device, size=16)

        captured_buffer_lengths = []
        captured_buffer_devices = []
        original_empty = torch.empty

        def empty_spy(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper around the real torch.empty that records the first
            # argument and the device keyword of every call, then forwards to the real allocator.
            captured_buffer_lengths.append(args[0])
            captured_buffer_devices.append(kwargs["device"])
            return original_empty(*args, **kwargs)

        with patch.object(
            benchmarker,
            "get_event_pairs",
            wraps=benchmarker.get_event_pairs,
        ) as mock_get_event_pairs:
            with patch.object(torch.version, "hip", hip_value):
                with patch(
                    "torch._inductor.runtime.benchmarking.torch.empty",
                    side_effect=empty_spy,
                ):
                    timing = benchmarker.benchmark_gpu(
                        _callable,
                        rep=1,
                        estimation_iters=1,
                        memory_warmup_iters=0,
                    )

        self.assertGreater(timing, 0)
        mock_get_event_pairs.assert_called_once_with(1, device_type=device)
        self.assertGreater(len(captured_buffer_lengths), 0)
        self.assertEqual(captured_buffer_lengths[0], expected_buffer_size_bytes // 4)
        self.assertEqual(captured_buffer_devices[0], device)


if __name__ == "__main__":
    run_tests()
