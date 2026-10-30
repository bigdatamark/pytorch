# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Tests for CUPTI PM sampling via the PmSampler class. Defines a fixed tuple of
# four single-pass metric names, a module flag that is true only with CUDA, libcupti 13.3 or
# newer and a non-empty supported metric list, and one test class gated on that flag. Importing
# the pm_sampling module is done conditionally at import time, only when cupti-python is
# present.
"""Tests for CUPTI PM sampling (``torch.profiler._cupti.pm_sampling``).

Covers the PmSampler sizing (decode-image = process-wide look-back / interval, capped), metric
selection + non-empty enforcement, and discovery -- exercised through real CUPTI PM sampling
(needs a perfmon-capable GPU; the tests skip otherwise).
"""

from __future__ import annotations

import time
import unittest

import torch
from torch.testing._internal.common_cuda import (
    TEST_CUDA,
    TEST_CUPTI as TEST_CUPTI_PYTHON,
    TEST_CUPTI_V13_3,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# pm_sampling imports the cupti module at load, so only import it when cupti-python is present;
# every test that uses these is gated on TEST_CUPTI_PM_SAMPLING (which implies TEST_CUPTI_PYTHON).
if TEST_CUPTI_PYTHON:
    from torch.profiler._cupti.pm_sampling import (
        _DEFAULT_LOOKBACK_WINDOW_NS,
        _DEFAULT_SAMPLING_INTERVAL_NS,
        PmSampler,
        supported_metrics,
    )


# Metrics are per-consumer (PmSampler.add_consumer), with no built-in default, so tests pass an
# explicit set (single-pass on the default 4-counter chips).
_TEST_METRICS = (
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "nvlrx__bytes.avg.pct_of_peak_sustained_elapsed",
    "nvltx__bytes.avg.pct_of_peak_sustained_elapsed",
)


# Needs CUDA + libcupti >= 13.3 (cupti-python ships with PyTorch) AND the nvperf profiler-host
# libraries (libnvperf_host/target): supported_metrics() returns empty (and logs) when they can't
# load, so it doubles as the availability probe (@cache, so it runs once). Whether the HW can
# actually engage a live session (perfmon access) is only known at start(), so the session tests
# still skip at runtime when none comes up.
TEST_CUPTI_PM_SAMPLING = TEST_CUDA and TEST_CUPTI_V13_3 and bool(supported_metrics())


@unittest.skipIf(
    not TEST_CUPTI_PM_SAMPLING,
    "requires cupti pm_sampling + the nvperf libraries + a capable CUDA GPU",
)
class TestPmSamplingWindowSizing(TestCase):
    # [CLASS DOC] Tests PmSampler window sizing, metric validation, configuration, consumer
    # sharing and retention against real CUPTI PM sampling. Tests that need a live session skip
    # at runtime when the sampler has no collector (perfmon unavailable) or no samples arrive.
    # Consumers are detached through cleanup hooks.
    """PM-sampling window sizing exercised through real CUPTI PM sampling: a PmSampler configured
    with a window + interval samples live GPU work, and the decoded frames confirm the sizing --
    max_samples = window // interval flows through configure()/decode(), each frame carries a column
    per metric, HW timestamps are monotonic, and the decoded span stays within the requested window.
    Skips at runtime when PM sampling cannot engage on the host (needs perfmon-capable HW)."""

    def _run_gpu_work(self, seconds: float = 0.2) -> None:
        # [FUNCTION DOC] HELPER: Busy-loops matrix multiply plus relu on a 512 by 512 CUDA
        # tensor until the given number of seconds (default 0.2) has elapsed, then synchronises
        # the device so the sampler sees real GPU activity.
        a = torch.randn(512, 512, device="cuda")
        deadline = time.time() + seconds
        while time.time() < deadline:
            a = torch.relu(a @ a)
        torch.cuda.synchronize()

    def _collect(self, metrics=_TEST_METRICS) -> tuple[PmSampler, list]:
        # [FUNCTION DOC] HELPER: Creates a PmSampler for the current device, registers a
        # consumer for the given metric names (default the shared test set), and schedules its
        # detach as test cleanup. Skips the test if the sampler has no live collector. Runs GPU
        # work, polls the handle once, and returns the sampler plus a list holding the frame, or
        # an empty list when no frame came back.
        # Real session: add_consumer enables PM sampling on the current device via CUPTI, we drive
        # some GPU work, and poll(handle) returns a frame of the ring's samples (raw HW-ns).
        sampler = PmSampler(torch.cuda.current_device())
        handle = sampler.add_consumer(list(metrics))
        self.addCleanup(handle.detach)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        self._run_gpu_work()
        frame = handle.poll()
        return sampler, ([frame] if frame is not None else [])

    def test_max_samples_from_process_config(self):
        # [FUNCTION DOC] Starts a real session with default process configuration and asserts
        # the sampler max sample count equals the default look-back window divided by the
        # default sampling interval, never less than one.
        # max_samples = look-back / interval (the process-wide default here, since no
        # configure() call); accepted by a real start()/decode() (which sizes the ring
        # from it via get_counter_data_size).
        sampler, _ = self._collect()
        expected = max(1, _DEFAULT_LOOKBACK_WINDOW_NS // _DEFAULT_SAMPLING_INTERVAL_NS)
        self.assertEqual(sampler._max_samples, expected)

    def test_configure_and_get_config_first_come_first_serve(self):
        # [FUNCTION DOC] Checks the process-wide configuration is first come first served. Saves
        # the class-level config and restores it in cleanup, resets to defaults, checks the
        # defaults report as not configured, configures 0.5 ms interval and 2000 ms look-back
        # and checks values in nanoseconds and the configured flag. A second configure call must
        # log a warning and leave the interval unchanged. Needs no CUDA work.
        # Interval/look-back are a process-wide config (no env var): configure() sets them,
        # get_config() reports them, and it's first-come-first-serve -- a second call is ignored.
        # No CUDA needed; save/restore the class state so other tests are unperturbed.
        saved = PmSampler.get_config()

        def _restore():
            # [FUNCTION DOC] HELPER: Cleanup callback that writes the saved interval, look-back
            # window and configured flag back onto the PmSampler class attributes.
            PmSampler._sampling_interval_ns = saved["sampling_interval_ns"]
            PmSampler._lookback_window_ns = saved["lookback_window_ns"]
            PmSampler._configured = saved["configured"]

        self.addCleanup(_restore)
        PmSampler._sampling_interval_ns = _DEFAULT_SAMPLING_INTERVAL_NS
        PmSampler._lookback_window_ns = _DEFAULT_LOOKBACK_WINDOW_NS
        PmSampler._configured = False
        self.assertEqual(
            PmSampler.get_config(),
            {
                "sampling_interval_ns": _DEFAULT_SAMPLING_INTERVAL_NS,
                "lookback_window_ns": _DEFAULT_LOOKBACK_WINDOW_NS,
                "configured": False,
            },
        )
        PmSampler.configure(sampling_interval_ms=0.5, lookback_window_ms=2000)
        cfg = PmSampler.get_config()
        self.assertEqual(cfg["sampling_interval_ns"], 500_000)
        self.assertEqual(cfg["lookback_window_ns"], 2_000_000_000)
        self.assertTrue(cfg["configured"])
        # first-come-first-serve: a second configure() is ignored (and warns).
        with self.assertLogs("torch.profiler._cupti.pm_sampling", level="WARNING"):
            PmSampler.configure(sampling_interval_ms=0.1)
        self.assertEqual(PmSampler.get_config()["sampling_interval_ns"], 500_000)

    def test_empty_metrics_rejected(self):
        # [FUNCTION DOC] Adding a consumer with an empty metric set must raise a ValueError, and
        # the sampler must not have created a collector.
        # A consumer must bring a non-empty metric set; add_consumer rejects an empty one.
        sampler = PmSampler(torch.cuda.current_device())
        with self.assertRaises(ValueError):
            sampler.add_consumer(())
        self.assertIsNone(sampler._col)

    def test_multipass_metrics_rejected(self):
        # [FUNCTION DOC] A metric needing multiple passes (a throughput metric) must make
        # add_consumer raise a ValueError whose message mentions passes, with no collector
        # created.
        # PM sampling is single-pass only; a metric that needs multiple passes (sm__throughput.*
        # needs ~17) is rejected by add_consumer -> ValueError, no session, running state untouched.
        # The class gate guarantees the nvperf profiler host is loadable, so the pass count is known.
        sampler = PmSampler(torch.cuda.current_device())
        with self.assertRaises(ValueError) as cm:
            sampler.add_consumer(["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
        self.assertIn("pass", str(cm.exception).lower())
        self.assertIsNone(sampler._col)

    def test_returns_per_device_singleton(self):
        # [FUNCTION DOC] PmSampler construction is a per-device singleton: constructing twice
        # for the current device, and constructing with no device argument, all return the
        # identical object.
        # PmSampler(device) returns the one instance for that device -- repeated construction and the
        # no-arg (current device) form resolve to the same object, so all consumers share one session.
        dev = torch.cuda.current_device()
        self.assertIs(PmSampler(dev), PmSampler(dev))
        self.assertIs(PmSampler(dev), PmSampler())

    def test_profiler_refcount_balanced(self):
        # [FUNCTION DOC] Records the class-wide active collector count, adds a consumer
        # (skipping after detaching if no collector started), asserts the count rose by one,
        # then detaches and asserts it returns to the starting value.
        # The process-global profiler deinit is refcounted (not by CUPTI): a full add/remove cycle
        # returns the live-collector count to where it started, so the profiler is torn down exactly
        # once and can be re-initialized on the next session.
        before = PmSampler._active
        sampler = PmSampler(torch.cuda.current_device())
        handle = sampler.add_consumer(list(_TEST_METRICS))
        if sampler._col is None:
            handle.detach()
            self.skipTest("PM sampling could not start on this GPU")
        self.assertEqual(PmSampler._active, before + 1)
        handle.detach()
        self.assertEqual(PmSampler._active, before)

    def test_decoded_frames_have_per_metric_columns(self):
        # [FUNCTION DOC] Collects frames from a live session and, if any, checks each contains
        # start_ns and device_id columns plus one column for every test metric name. Skips when
        # no frames are produced.
        # Each sampled metric is a value column keyed by its metric name (self-describing frame).
        _, frames = self._collect()
        if not frames:
            self.skipTest("no PM samples produced on this GPU")
        for f in frames:
            self.assertIn("start_ns", f)
            self.assertIn("device_id", f)
            for name in _TEST_METRICS:
                self.assertIn(name, f)

    def test_decoded_span_within_lookback(self):
        # [FUNCTION DOC] Concatenates start timestamps from collected frames (skips if none),
        # checks the sorted timestamps are non-decreasing and that the span from earliest to
        # latest does not exceed the default look-back window.
        import numpy as np

        _, frames = self._collect()
        ts = (
            np.concatenate([f["start_ns"] for f in frames])
            if frames
            else np.empty(0, dtype=np.int64)
        )
        if ts.size == 0:
            self.skipTest("no PM samples produced on this GPU")
        self.assertTrue((np.diff(np.sort(ts)) >= 0).all())  # monotonic HW timestamps
        self.assertLessEqual(int(ts.max() - ts.min()), _DEFAULT_LOOKBACK_WINDOW_NS)

    def test_select_metrics_by_name(self):
        # [FUNCTION DOC] Collects with only the first two test metrics. Asserts the sampler
        # metric name list equals the chosen list, and that frames contain those columns but not
        # the third (unselected) metric.
        # A caller passes metric name strings; each selected metric becomes its own frame column.
        chosen = list(_TEST_METRICS[:2])
        sampler, frames = self._collect(metrics=chosen)
        self.assertEqual(sampler._metric_names, chosen)
        if not frames:
            self.skipTest("no PM samples produced on this GPU")
        for f in frames:
            for name in chosen:
                self.assertIn(name, f)
            self.assertNotIn(_TEST_METRICS[2], f)  # unselected

    def test_supported_metrics_lists_chip_counters(self):
        # [FUNCTION DOC] Calls supported_metrics, skips when empty, asserts every entry is a
        # string and that the base names (text before the first dot) include the SM cycles
        # active counter.
        metrics = supported_metrics()
        if not metrics:
            self.skipTest("profiler host could not enumerate metrics on this chip")
        self.assertTrue(all(isinstance(m, str) for m in metrics))
        base = {m.split(".", 1)[0] for m in metrics}
        self.assertIn(
            "sm__cycles_active", base
        )  # our default SM counter is in the menu

    def test_unknown_metric_warns(self):
        # [FUNCTION DOC] Skips if no supported metrics are listed. Adds a consumer asking for a
        # made-up metric name and asserts a warning is logged by the pm_sampling logger
        # containing the text saying the chip does not report it. Detach is registered as
        # cleanup.
        # A metric the chip does not report is warned about (not silently accepted).
        supported = supported_metrics()
        if not supported:
            self.skipTest("profiler host could not enumerate metrics on this chip")
        sampler = PmSampler(torch.cuda.current_device())
        with self.assertLogs(
            "torch.profiler._cupti.pm_sampling", level="WARNING"
        ) as cm:
            handle = sampler.add_consumer(["not__a_real_metric.avg"])
            self.addCleanup(handle.detach)
        self.assertTrue(any("not reported by this chip" in m for m in cm.output))

    def test_multiple_consumers_share_session(self):
        # [FUNCTION DOC] Two consumers on one device with different metrics share one session.
        # Asserts the sampler metric list is the ordered union, then polls both and checks each
        # frame holds only its own consumer's metric columns. Skips when no collector or no
        # samples.
        # Two consumers on one device share the single session: it samples the union of their
        # metrics, and each consumer's polled frame carries only its own metric columns.
        a_metrics, b_metrics = [_TEST_METRICS[0]], list(_TEST_METRICS[1:3])
        sampler = PmSampler(torch.cuda.current_device())
        a = sampler.add_consumer(a_metrics)
        self.addCleanup(a.detach)
        b = sampler.add_consumer(b_metrics)
        self.addCleanup(b.detach)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        # The session samples the union (dedup, order-preserving).
        self.assertEqual(sampler._metric_names, a_metrics + b_metrics)
        self._run_gpu_work()
        fa = a.poll()
        fb = b.poll()
        if fa is None or fb is None:
            self.skipTest("no PM samples produced on this GPU")
        self.assertIn(a_metrics[0], fa)  # only consumer A's metric
        self.assertNotIn(b_metrics[0], fa)
        for name in b_metrics:  # only consumer B's metrics
            self.assertIn(name, fb)
        self.assertNotIn(a_metrics[0], fb)

    def test_metric_union_reconfigure(self):
        # [FUNCTION DOC] Adds a second consumer with a distinct metric so the union grows
        # (session rebuilt), checks the names list and that the first consumer still polls
        # frames with its metric. Detaching the second shrinks the union back and sampling must
        # still work, with a non-empty frame containing the first metric.
        # Adding/removing a consumer with a distinct metric changes the union and rebuilds the
        # session (a live in-place reconfigure segfaults); sampling must survive each grow/shrink.
        m0, m1 = [_TEST_METRICS[0]], [_TEST_METRICS[1]]
        sampler = PmSampler(torch.cuda.current_device())
        a = sampler.add_consumer(m0)
        self.addCleanup(a.detach)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        self.assertEqual(sampler._metric_names, m0)
        b = sampler.add_consumer(m1)  # union grows -> rebuild
        self.addCleanup(b.detach)
        self.assertEqual(sampler._metric_names, m0 + m1)
        self._run_gpu_work()
        fa = a.poll()
        if fa is None:
            self.skipTest("no PM samples produced on this GPU")
        self.assertIn(m0[0], fa)  # sampling works after the grow-rebuild
        b.detach()  # union shrinks -> rebuild
        self.assertEqual(sampler._metric_names, m0)
        self._run_gpu_work()
        fa = a.poll()
        self.assertTrue(
            fa is not None and m0[0] in fa
        )  # works after the shrink-rebuild

    def test_dropped_handle_detaches(self):
        # [FUNCTION DOC] Drops the only reference to a handle without detaching, runs garbage
        # collection, and asserts the sampler consumer count goes from one to zero, showing the
        # handle destructor unregisters it.
        # Safety net: dropping a handle without detach() still unregisters it -- the sampler holds
        # only the _Consumer record, not the handle, so the handle is collectable and __del__ fires.
        import gc

        sampler = PmSampler(torch.cuda.current_device())
        handle = sampler.add_consumer(list(_TEST_METRICS))
        if sampler._col is None:
            handle.detach()
            self.skipTest("PM sampling could not start on this GPU")
        self.assertEqual(len(sampler._consumers), 1)
        handle = None  # drop the only reference
        gc.collect()
        self.assertEqual(len(sampler._consumers), 0)  # __del__ detached it

    def test_retention_only_for_lagging_consumer(self):
        # [FUNCTION DOC] Checks sample retention: a lone consumer that has polled leaves zero
        # retained timestamps. After a second consumer with the same metric is added, a poll by
        # the first must leave retained timestamps for the lagging second, which then polls a
        # non-empty frame from that buffer.
        # decode drains, so the sampler retains samples for a consumer that hasn't polled yet, and
        # keeps nothing for a lone, caught-up consumer (GC'd by the min-cursor watermark).
        sampler = PmSampler(torch.cuda.current_device())
        a = sampler.add_consumer(list(_TEST_METRICS[:1]))
        self.addCleanup(a.detach)
        if sampler._col is None:
            self.skipTest("PM sampling could not start on this GPU")
        # Lone consumer: after it polls, nothing is retained (it consumed everything it drained).
        self._run_gpu_work()
        if a.poll() is None:
            self.skipTest("no PM samples produced on this GPU")
        self.assertEqual(sampler._retained_ts.size, 0)
        # Add a second consumer (same metric -> no reconfigure); a poll by A must now retain A's
        # drained samples for B, which lags -- and B gets them from the buffer (A already drained
        # the HW ring).
        b = sampler.add_consumer(list(_TEST_METRICS[:1]))
        self.addCleanup(b.detach)
        self._run_gpu_work()
        a.poll()
        self.assertGreater(sampler._retained_ts.size, 0)
        fb = b.poll()
        self.assertTrue(fb is not None and fb["start_ns"].size > 0)


if __name__ == "__main__":
    run_tests()
