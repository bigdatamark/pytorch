# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Tests that the Kineto profiler captures events from C++ threads driven by a
# native training engine (CUDA and XPU only). IMPORT-TIME SIDE EFFECTS: unless running in
# fbcode it temporarily changes the working directory to this file's directory and builds and
# loads a C++ extension named profiler_test_cpp_thread_lib from the neighbouring C++ source
# (verbose build output), then restores the directory; in fbcode it imports a prebuilt
# library. Defines module globals for the active profiler, iteration counts and device name, a
# Python subclass of the native event handler, and a device-generic test class instantiated
# for cuda and xpu.

import os

import torch
import torch.utils.cpp_extension
from torch._environment import is_fbcode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCUDAIf,
    skipXPUIf,
)
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


if is_fbcode():
    import caffe2.test.profiler_test_cpp_thread_lib as cpp  # @manual=//caffe2/test:profiler_test_cpp_thread_lib
else:
    # cpp extensions use relative paths. Those paths are relative to
    # this file, so we'll change the working directory temporarily
    old_working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cpp = torch.utils.cpp_extension.load(
        name="profiler_test_cpp_thread_lib",
        sources=[
            "test_cpp_thread.cpp",
        ],
        verbose=True,
    )

    # return the working directory (see setUp)
    os.chdir(old_working_dir)


KinetoProfiler = None
IterationCount = 5
ActivateIteration = 2
device = None


def blueprint(text):
    # [FUNCTION DOC] HELPER: Prints the text to standard output wrapped in terminal color
    # escape codes for blue.
    print(f"\33[34m{text}\33[0m")


# onIterationStart() will be called by C++ training engine in cpp_thread_test_lib.cpp
class PythonProfilerEventHandler(cpp.ProfilerEventHandler):
    # [CLASS DOC] Python implementation of the native profiler event handler interface. The
    # C++ training engine calls these methods from its worker threads: onIterationStart once
    # per iteration to control the module-level profiler, and emulateTraining to run the
    # workload.
    def onIterationStart(self, iteration: int) -> None:
        # [FUNCTION DOC] Called by the native engine at the start of each iteration (always on
        # the same thread). Uses the module-level profiler: iteration zero starts it, the last
        # iteration (as set by the iteration count global) stops it, and every iteration in
        # between steps it. Prints a colored status line each time. The profiler must be
        # started on the thread that steps it.
        global KinetoProfiler, IterationCount
        # it is important to start the profiler on the same thread that step() is called
        # and yes, onIterationStart() will always be called on the same thread
        if iteration == 0:
            # this also means step() starts on iteration 1, not 0
            KinetoProfiler.start()
            blueprint("starting kineto profiler")
        elif iteration == IterationCount - 1:
            KinetoProfiler.stop()
            blueprint("stopping kineto profiler")
        else:
            blueprint("stepping kineto profiler")
            KinetoProfiler.step()

    def emulateTraining(self, iteration: int, thread_id: int) -> None:
        # [FUNCTION DOC] Called by the native engine on worker threads to emulate a training
        # step. Looks up the device module named by the global device string and raises
        # AssertionError if it has no synchronize function. Inside a user_function record
        # scope it adds two one-element tensors on that device, copies the result to cpu and
        # synchronizes the device. The iteration and thread id arguments are unused.
        global device
        # blueprint(f"training iteration {iteration} in thread {thread_id}")
        torch_device = getattr(torch, device)
        if not hasattr(torch_device, "synchronize"):
            raise AssertionError(f"Device {device} does not have synchronize method")
        sync_func = torch_device.synchronize

        with torch.autograd.profiler.record_function("user_function"):
            a = torch.ones(1, device=device)
            b = torch.ones(1, device=device)
            torch.add(a, b).cpu()
            sync_func()


class CppThreadTest(TestCase):
    # [CLASS DOC] Device-type test case (instantiated for cuda and xpu) that checks which
    # events the profiler records when the native engine runs workloads in child threads, with
    # or without profiling enabled in those threads. Class attributes: the worker thread
    # count, the shared event handler instance and the latest trace object delivered by the
    # profiler callback.
    ThreadCount = 20  # set to 2 for debugging
    EventHandler = None
    TraceObject = None

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] Creates one Python event handler instance, stores it on the class and
        # registers it with the native extension as the active handler.
        super().setUpClass()
        cls.EventHandler = PythonProfilerEventHandler()
        cpp.ProfilerEventHandler.Register(cls.EventHandler)

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Removes the cpp extension build root unless running in fbcode. Does
        # not call the parent tearDownClass.
        if not is_fbcode():
            torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    def setUp(self) -> None:
        # [FUNCTION DOC] Records the device type of this test into the global device name.
        # Builds a profiler without memory profiling and runs the native thread starter with
        # one thread and no profiling in the thread, so that initialization events are
        # generated and cleared out.
        super().setUp()
        global device
        device = self.device_type

        # this clears off events from initialization
        self.start_profiler(False)
        cpp.start_threads(1, IterationCount, False)

    def start_profiler(self, profile_memory):
        # [FUNCTION DOC] Creates a profiler with a wait one, warmup one, active (global
        # activate iteration count) and repeat once schedule, stack and shape recording, the
        # given memory profiling flag, and this test's trace callback. Stores it in the global
        # profiler variable without starting it.
        global KinetoProfiler
        KinetoProfiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=ActivateIteration, repeat=1
            ),
            on_trace_ready=self.set_trace,
            with_stack=True,
            profile_memory=profile_memory,
            record_shapes=True,
        )

    def set_trace(self, trace_obj) -> None:
        # [FUNCTION DOC] Profiler trace-ready callback: stores the given trace object on the
        # class so that later checks can read its events.
        type(self).TraceObject = trace_obj

    def assert_text(self, condition, text, msg):
        # [FUNCTION DOC] HELPER: Prints the text in green when the condition is true and red
        # otherwise, then asserts the condition with the given failure message.
        if condition:
            print(f"\33[32m{text}\33[0m")
        else:
            print(f"\33[31m{text}\33[0m")
        self.assertTrue(condition, msg)

    def check_trace(self, expected, mem=False) -> None:
        # [FUNCTION DOC] Checks the events of the stored trace against an expected mapping
        # from event name to a pair of expected count and device type name. The minimum count
        # is the expected count times one less than the activate iteration count. In memory
        # mode, counts events of that name and device whose printed form shows a nonzero
        # device memory usage field and requires at least the minimum. Otherwise, when the
        # expected count is one (baseline without child profiling) requires the event count to
        # equal one times the activate iteration count; other counts require at least the
        # minimum. Failures raise AssertionError through assert_text.
        blueprint("verifying trace")
        event_list = type(self).TraceObject.events()
        for key, values in expected.items():
            count = values[0]
            min_count = count * (ActivateIteration - 1)
            dev = values[1]
            filtered = filter(
                lambda ev: ev.name == key
                and str(ev.device_type) == f"DeviceType.{dev}",
                event_list,
            )

            if mem:
                mem_key = f"{self.device_type}_memory_usage"
                actual = 0
                for ev in filtered:
                    sev = str(ev)
                    has_device_memory_usage = (
                        sev.find(f"{mem_key}=0 ") < 0 and sev.find(f"{mem_key}=") > 0
                    )
                    if has_device_memory_usage:
                        actual += 1
                self.assert_text(
                    actual >= min_count,
                    f"{key}: {actual} >= {min_count}",
                    f"not enough event with {mem_key} set",
                )
            else:
                actual = len(list(filtered))
                if count == 1:  # test_without
                    count *= ActivateIteration
                    self.assert_text(
                        actual == count,
                        f"{key}: {actual} == {count}",
                        "baseline event count incorrect",
                    )
                else:
                    self.assert_text(
                        actual >= min_count,
                        f"{key}: {actual} >= {min_count}",
                        "not enough event recorded",
                    )

    @skipCUDAIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    @skipXPUIf(
        True,
        "The XPU Profiler will not cover this case for now. Will support it in next period.",
    )
    def test_with_enable_profiler_in_child_thread(self, device) -> None:
        # [FUNCTION DOC] Runs the native threads with profiling enabled in each child thread and
        # checks the trace has at least the thread count worth of add events on cpu and
        # user_function events on the test device. Skipped on Windows CUDA and always skipped on
        # XPU.
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
                "user_function": [self.ThreadCount, self.device_type.upper()],
            }
        )

    @skipCUDAIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    @skipXPUIf(
        True,
        "The XPU Profiler will not cover this case for now. Will support it in next period.",
    )
    def test_without_enable_profiler_in_child_thread(self, device) -> None:
        # [FUNCTION DOC] Baseline: runs the native threads without profiling enabled in child
        # threads and checks the trace has exactly the activate iteration count of add events on
        # cpu and of user_function events on the test device. Skipped on Windows CUDA and always
        # skipped on XPU.
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, False)
        self.check_trace(
            {
                "aten::add": [1, "CPU"],
                "user_function": [1, self.device_type.upper()],
            }
        )

    @skipCUDAIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    @skipXPUIf(
        True,
        "The XPU Profiler will not cover this case for now. Will support it in next period.",
    )
    def test_profile_memory(self, device) -> None:
        # [FUNCTION DOC] Starts the profiler with memory profiling, runs native threads with child
        # thread profiling, and checks there are enough add events on cpu carrying nonzero memory
        # usage. Skipped on Windows CUDA and always skipped on XPU.
        self.start_profiler(True)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
            },
            mem=True,
        )


instantiate_device_type_tests(CppThreadTest, globals(), only_for=("cuda", "xpu"))

if __name__ == "__main__":
    run_tests()
