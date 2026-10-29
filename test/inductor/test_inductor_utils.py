# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for benchmarking utilities in torch._inductor.utils: the plain GPU
# benchmarker, do_bench_using_profiling, profiler-based device time extraction, and fused kernel
# naming. Defines fake Kineto and profiler event classes so profiling logic runs without a real
# profiler. Import-time side effect: picks the current accelerator type, or cpu, as device_type.

import functools
import logging
from unittest import mock

import torch
from torch._inductor import utils
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import do_bench_using_profiling
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)


class FakeKinetoEvent:
    # [CLASS DOC] HELPER: Minimal stand-in for a Kineto profiler event. Stores the name, device
    # type, start and end timestamps in nanoseconds, correlation ids and activity type given at
    # construction and returns each through a same-named accessor method, as the real event
    # objects do.
    def __init__(
        self,
        name: str,
        device_type: DeviceType,
        start_ns: int,
        end_ns: int,
        linked_correlation_id: int = 0,
        correlation_id: int = 0,
        activity_type: str = "kernel",
    ) -> None:
        # [FUNCTION DOC] HELPER: Store all constructor arguments as private attributes; no
        # validation.
        self._name = name
        self._device_type = device_type
        self._start_ns = start_ns
        self._end_ns = end_ns
        self._linked_correlation_id = linked_correlation_id
        self._correlation_id = correlation_id
        self._activity_type = activity_type

    def name(self) -> str:
        # [FUNCTION DOC] HELPER: Return the event name.
        return self._name

    def device_type(self) -> DeviceType:
        # [FUNCTION DOC] HELPER: Return the device type of the event.
        return self._device_type

    def start_ns(self) -> int:
        # [FUNCTION DOC] HELPER: Return the start timestamp in nanoseconds.
        return self._start_ns

    def end_ns(self) -> int:
        # [FUNCTION DOC] HELPER: Return the end timestamp in nanoseconds.
        return self._end_ns

    def linked_correlation_id(self) -> int:
        # [FUNCTION DOC] HELPER: Return the correlation id of the CPU event this event links to
        # (0 if none).
        return self._linked_correlation_id

    def correlation_id(self) -> int:
        # [FUNCTION DOC] HELPER: Return the event's own correlation id.
        return self._correlation_id

    def activity_type(self) -> str:
        # [FUNCTION DOC] HELPER: Return the activity type string (default kernel).
        return self._activity_type


class FakeProfilerEvent:
    # [CLASS DOC] HELPER: Minimal stand-in for a profiler function event, exposing name, device
    # type, integer id and a list of CPU child events as plain attributes.
    def __init__(
        self,
        name: str,
        device_type: DeviceType,
        id: int,
        cpu_children: list["FakeProfilerEvent"] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Store name, device type and id; the children list defaults to
        # empty.
        self.name = name
        self.device_type = device_type
        self.id = id
        self.cpu_children = cpu_children or []


class TestBench(TestCase):
    # [CLASS DOC] Benchmarking tests. The class setup builds a half precision linear layer
    # benchmark callable on the accelerator (or cpu) and shares it with the tests through a
    # class attribute.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # [FUNCTION DOC] Class setup: create random half precision input and weight tensors on
        # device_type and store a partial application of the linear function over them as a
        # static class attribute.
        x = torch.rand(1024, 10).to(device_type).half()
        w = torch.rand(512, 10).to(device_type).half()
        cls._bench_fn = staticmethod(
            functools.partial(torch.nn.functional.linear, x, w)
        )

    def test_benchmarker(self):
        # [FUNCTION DOC] Run the standard GPU benchmarker on the shared callable, log the
        # result, and check that the reported time is positive.
        res = benchmarker.benchmark_gpu(self._bench_fn)
        log.warning("do_bench result: %s", res)
        self.assertGreater(res, 0)

    def test_do_bench_using_profiling(self):
        # [FUNCTION DOC] Run the profiling-based benchmark on the shared callable, log it, and
        # check that the reported time is positive.
        res = do_bench_using_profiling(self._bench_fn)
        log.warning("do_bench_using_profiling result: %s", res)
        self.assertGreater(res, 0)

    def test_do_bench_profile_result_uses_linked_device_events(self):
        # [FUNCTION DOC] Feed fake profiler data with one benchmark event with three CPU
        # children. Two kernels are linked to children by linked id, a third is matched through
        # its own correlation id since its link id is zero, and the annotation event is ignored.
        # Asserts the result is about 0.003 ms: three kernels of 1 microsecond each, divided by
        # one repeat.
        profiler_events = [
            FakeProfilerEvent(
                utils._DO_BENCH_PROFILE_EVENT_NAME,
                DeviceType.CPU,
                1,
                [
                    FakeProfilerEvent("aten::first", DeviceType.CPU, 2),
                    FakeProfilerEvent("aten::second", DeviceType.CPU, 3),
                    FakeProfilerEvent("cudaGraphLaunch", DeviceType.CPU, 4),
                ],
            )
        ]
        kineto_events = [
            FakeKinetoEvent(
                utils._DO_BENCH_PROFILE_EVENT_NAME,
                DeviceType.CUDA,
                0,
                11000,
                correlation_id=1,
                activity_type="gpu_user_annotation",
            ),
            FakeKinetoEvent(
                "first_kernel",
                DeviceType.CUDA,
                0,
                1000,
                linked_correlation_id=2,
            ),
            FakeKinetoEvent(
                "second_kernel",
                DeviceType.CUDA,
                10000,
                11000,
                linked_correlation_id=3,
            ),
            FakeKinetoEvent(
                "cuda_graph_kernel",
                DeviceType.CUDA,
                20000,
                21000,
                correlation_id=4,
            ),
        ]

        self.assertAlmostEqual(
            utils._get_do_bench_profile_result(
                kineto_events, profiler_events, 1, DeviceType.CUDA
            ),
            0.003,
        )

    def test_get_fused_kernel_name_windows_truncation(self):
        # [FUNCTION DOC] Check get_fused_kernel_name with a node schedule whose node has ten
        # origins with long names. With the Windows check patched to false the name must exceed
        # 50 characters. With it patched to true the name must be at most 50 characters, start
        # with the fused prefix and end in an eight character hash suffix.
        class FakeOrigin:
            # [CLASS DOC] HELPER: Fake FX origin node with a name and a call_function op.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Store the name and set the op to call_function.
                self.name = name
                self.op = "call_function"

        class FakeIRNode:
            # [CLASS DOC] HELPER: Fake IR node holding only an origins collection.
            def __init__(self, origins):
                # [FUNCTION DOC] HELPER: Store the origins attribute.
                self.origins = origins

        class FakeSchedulerNode:
            # [CLASS DOC] HELPER: Fake scheduler node wrapping a fake IR node built from the
            # given origins.
            def __init__(self, origins):
                # [FUNCTION DOC] HELPER: Wrap a fake IR node built from the origins as the node
                # attribute.
                self.node = FakeIRNode(origins)

        origins = OrderedSet(
            FakeOrigin(f"some_very_long_op_name_that_exceeds_the_limit_{i}")
            for i in range(10)
        )
        node_schedule = [FakeSchedulerNode(origins)]

        # On non-Windows the full descriptive name is kept.
        with mock.patch("torch._inductor.utils.is_windows", return_value=False):
            name = utils.get_fused_kernel_name(node_schedule, "inductor_node")
            self.assertGreater(len(name), 50)

        # On Windows the name is truncated and a hash suffix is appended.
        with mock.patch("torch._inductor.utils.is_windows", return_value=True):
            name_win = utils.get_fused_kernel_name(node_schedule, "inductor_node")
            self.assertLessEqual(len(name_win), 50)
            self.assertTrue(name_win.startswith("fused_"))
            self.assertEqual(len(name_win.rsplit("_", 1)[-1]), 8)

    def test_do_bench_profile_result_requires_record_function_event(self):
        # [FUNCTION DOC] Verify that when profiler events contain a benchmark event but no
        # kineto device event is linked to it, the profile result helper raises RuntimeError
        # mentioning a capture failure.
        with self.assertRaisesRegex(RuntimeError, "Failed to capture"):
            utils._get_do_bench_profile_result(
                [FakeKinetoEvent("user_kernel", DeviceType.CUDA, 0, 1000)],
                [
                    FakeProfilerEvent(
                        utils._DO_BENCH_PROFILE_EVENT_NAME,
                        DeviceType.CPU,
                        1,
                    )
                ],
                1,
                DeviceType.CUDA,
            )


if __name__ == "__main__":
    run_tests(device_type)
