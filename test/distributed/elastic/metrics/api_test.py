#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Unit tests for torch.distributed.elastic.metrics.api: metric-name derivation
# for functions and methods, and the prof decorator emitting success/failure counters and
# a duration metric to a metric stream, including use on a subclass implementation of an
# abstract method. Defines small fixtures (a recording MetricHandler, Parent/Child classes).
# Runs via run_tests when executed as a script.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs
import abc
import unittest.mock as mock

from torch.distributed.elastic.metrics.api import (
    _get_metric_name,
    MetricData,
    MetricHandler,
    MetricStream,
    prof,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def foo_1():
    # [FUNCTION DOC] HELPER: No-op module-level function used as a name-derivation fixture.
    pass


class TestMetricsHandler(MetricHandler):
    # [CLASS DOC] HELPER: MetricHandler that records emitted metrics in a mapping from
    # metric name to the latest MetricData (later emits with the same name overwrite).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty metric_data mapping.
        self.metric_data = {}

    def emit(self, metric_data: MetricData):
        # [FUNCTION DOC] HELPER: Stores the given MetricData under its name, replacing any
        # previous entry with that name.
        self.metric_data[metric_data.name] = metric_data


class Parent(abc.ABC):
    # [CLASS DOC] HELPER: Abstract base fixture with an abstract func and a base_func that
    # delegates to it.
    @abc.abstractmethod
    def func(self):
        # [FUNCTION DOC] HELPER: Abstract hook; the base body raises NotImplementedError.
        raise NotImplementedError

    def base_func(self):
        # [FUNCTION DOC] HELPER: Calls the subclass func, so profiling on the override fires.
        self.func()


class Child(Parent):
    # [CLASS DOC] HELPER: Concrete fixture whose func override is decorated with prof.
    # need to decorate the implementation not the abstract method!
    @prof
    def func(self):
        # [FUNCTION DOC] HELPER: Profiled no-op implementation.
        pass


class MetricsApiTest(TestCase):
    # [CLASS DOC] Tests for _get_metric_name and the prof decorator, using the profiled
    # fixture methods bar, throw and bar2 defined on this class.
    def foo_2(self):
        # [FUNCTION DOC] HELPER: Undecorated no-op method used for name derivation.
        pass

    @prof
    def bar(self):
        # [FUNCTION DOC] HELPER: Profiled no-op that succeeds (default group).
        pass

    @prof
    def throw(self):
        # [FUNCTION DOC] HELPER: Profiled method that always raises RuntimeError.
        raise RuntimeError

    @prof(group="torchelastic")
    def bar2(self):
        # [FUNCTION DOC] HELPER: Profiled no-op that succeeds, with group torchelastic.
        pass

    def test_get_metric_name(self):
        # [FUNCTION DOC] Sets foo_1's module to "api_test" so the name is stable across
        # runners, then checks _get_metric_name gives "api_test.foo_1" for the function and
        # "MetricsApiTest.foo_2" for the bound method.
        # Note: since pytorch uses main method to launch tests,
        # the module will be different between fb and oss, this
        # allows keeping the module name consistent.
        foo_1.__module__ = "api_test"
        self.assertEqual("api_test.foo_1", _get_metric_name(foo_1))
        self.assertEqual("MetricsApiTest.foo_2", _get_metric_name(self.foo_2))

    def test_profile(self):
        # [FUNCTION DOC] With getStream patched to return a stream bound to a recording
        # handler, checks bar emits a success counter of 1 and a duration metric but no
        # failure; throw raises RuntimeError and emits failure 1 and a duration; and bar2
        # emits metrics carrying group name torchelastic.
        handler = TestMetricsHandler()
        stream = MetricStream("torchelastic", handler)
        # patch instead of configure to avoid conflicts when running tests in parallel
        with mock.patch(
            "torch.distributed.elastic.metrics.api.getStream", return_value=stream
        ):
            self.bar()

            self.assertEqual(1, handler.metric_data["MetricsApiTest.bar.success"].value)
            self.assertNotIn("MetricsApiTest.bar.failure", handler.metric_data)
            self.assertIn("MetricsApiTest.bar.duration.ms", handler.metric_data)

            with self.assertRaises(RuntimeError):
                self.throw()

            self.assertEqual(
                1, handler.metric_data["MetricsApiTest.throw.failure"].value
            )
            self.assertNotIn("MetricsApiTest.bar_raise.success", handler.metric_data)
            self.assertIn("MetricsApiTest.throw.duration.ms", handler.metric_data)

            self.bar2()
            self.assertEqual(
                "torchelastic",
                handler.metric_data["MetricsApiTest.bar2.success"].group_name,
            )

    def test_inheritance(self):
        # [FUNCTION DOC] Calls base_func on a Child with getStream patched; checks the
        # profiled override emits "Child.func.success" with value 1 and a duration metric,
        # named after the concrete class.
        handler = TestMetricsHandler()
        stream = MetricStream("torchelastic", handler)
        # patch instead of configure to avoid conflicts when running tests in parallel
        with mock.patch(
            "torch.distributed.elastic.metrics.api.getStream", return_value=stream
        ):
            c = Child()
            c.base_func()

            self.assertEqual(1, handler.metric_data["Child.func.success"].value)
            self.assertIn("Child.func.duration.ms", handler.metric_data)


if __name__ == "__main__":
    run_tests()
