# Owner(s): ["module: multithreading"]
# [MODULE DOC] Thread safety stress tests meant to be run under ThreadSanitizer or free-
# threaded Python. Each test hammers one shared piece of state (tensor storage wrapper,
# profiler, grad_fn wrapper, global config flags) from several threads using the
# run_concurrently helper.

import weakref
from functools import partial

import torch
from torch.testing._internal.common_utils import run_concurrently, run_tests, TestCase


class TestTSan(TestCase):
    # [CLASS DOC] Test case holding the concurrency stress tests; stateless. Failures show up
    # as crashes, sanitizer reports or assertion errors.
    def test_storage_thread_safety(self):
        # [FUNCTION DOC] Ten times: creates a small float tensor, has four threads each fetch
        # its untyped storage and return a weak reference to it, then checks every weak
        # reference still resolves to the tensor's current storage.
        # Concurrent calls to tensor.untyped_storage()
        def access_untyped_storage(tensor):
            # [FUNCTION DOC] HELPER: Worker run in each thread; returns a weak reference to
            # the untyped storage of the given tensor.
            return weakref.ref(tensor.untyped_storage())

        for _ in range(10):
            tensor = torch.tensor([1.0, 2.0, 3.0])
            weakrefs = run_concurrently(
                access_untyped_storage, args=(tensor,), num_threads=4
            )
            for wr in weakrefs:
                self.assertEqual(wr(), tensor.untyped_storage())

    def test_concurrent_profiling(self):
        # [FUNCTION DOC] Runs one thread that repeatedly (thirty times) starts and stops the
        # profiler with stack recording while eight other threads do dict, sorting and small
        # tensor work. Passes if nothing crashes; there are no explicit assertions.
        """Repeatedly start/stop profiling while background threads are active.

        On free-threaded Python (3.14t+), this exercises concurrent access to
        the profiler's per-thread state without GIL protection. Without the
        thread-safety fixes (setprofileAllThreads, per-thread ValueCache,
        StopTheWorldGuard), this crashes from heap corruption due to data
        races on the shared hash maps.
        """

        def work():
            # [FUNCTION DOC] HELPER: Background load for the profiling test; one hundred
            # iterations of building a small dict of lists, sorting its items by list length,
            # and adding two small tensors.
            for _ in range(100):
                d = {str(i): list(range(i % 10)) for i in range(20)}
                _ = sorted(d.items(), key=lambda x: len(x[1]))
                torch.ones(10) + torch.zeros(10)

        def profile_work():
            # [FUNCTION DOC] HELPER: Worker that enters and leaves a stack-recording profiler
            # context thirty times, creating a small tensor inside each.
            for _ in range(30):
                with torch.profiler.profile(with_stack=True):
                    torch.ones(10)

        run_concurrently([profile_work] + [work] * 8)

    def test_concurrent_grad_fn(self):
        # [FUNCTION DOC] Ten times: builds a tensor that requires grad and has a grad_fn, has
        # four threads read that grad_fn at once, and checks every thread got the identical
        # object.
        # Concurrent access to tensor.grad_fn exercises the atomic
        # PyObjectSlot on Node that lazily creates the Python wrapper.
        def access_grad_fn(tensor):
            # [FUNCTION DOC] HELPER: Worker returning the grad_fn attribute of the given
            # tensor.
            return tensor.grad_fn

        for _ in range(10):
            x = torch.randn(4, requires_grad=True)
            y = x * 2 + 1
            grad_fns = run_concurrently(access_grad_fn, args=(y,), num_threads=4)
            for gf in grad_fns:
                self.assertIs(gf, y.grad_fn)

    def test_concurrent_config_flag_toggles(self):
        # [FUNCTION DOC] Builds a list of setter and getter pairs for process-global boolean
        # flags: warn-always, the two vmap fallback flags (the warning one has no getter), and
        # the cudnn conv benchmark empty-cache flag only when the binding exists. For each
        # pair, four threads concurrently toggle the flag; the original value is restored
        # afterwards when a getter exists. No explicit assertions; it relies on sanitizers or
        # crashes.
        """Concurrently set/read process-global bool config flags.

        warn_always, the vmap fallback enable/warning flags, and (on CUDA
        builds) the cudnn conv benchmark empty-cache flag are process-global
        toggles accessed through dedicated setters/getters. They used to be
        plain ``static bool``, so concurrent access from multiple threads was a
        data race -- undefined behavior under the C++ memory model and reported
        by TSan even though no real platform tears a bool load. They are now
        ``static constinit std::atomic<bool>`` with relaxed ordering; this
        hammers each accessor from several threads to guard against regressing.
        """
        functorch = torch._C._functorch

        # (setter, getter): getter is None when no read accessor is exposed to
        # Python, in which case the writer-vs-writer race still exercises it.
        flags = [
            (torch._C._set_warnAlways, torch._C._get_warnAlways),
            (
                functorch._set_vmap_fallback_enabled,
                functorch._is_vmap_fallback_enabled,
            ),
            (functorch._set_vmap_fallback_warning_enabled, None),
        ]
        # CUDA-only binding; absent on the CPU-only TSan build.
        if hasattr(torch._C, "_cudnn_set_conv_benchmark_empty_cache"):
            flags.append(
                (
                    torch._C._cudnn_set_conv_benchmark_empty_cache,
                    torch._C._cuda_get_conv_benchmark_empty_cache,
                )
            )

        def toggle(setter, getter, idx):
            # [FUNCTION DOC] HELPER: Worker that calls the setter two hundred times with
            # alternating booleans (parity of the thread index plus the loop counter), and
            # also calls the getter each time when one is given.
            for i in range(200):
                setter(bool((idx + i) & 1))
                if getter is not None:
                    getter()

        for setter, getter in flags:
            original = getter() if getter is not None else None
            try:
                run_concurrently([partial(toggle, setter, getter, t) for t in range(4)])
            finally:
                if original is not None:
                    setter(original)


if __name__ == "__main__":
    run_tests()
